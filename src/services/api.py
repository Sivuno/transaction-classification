import json
import logging
import random
import asyncio
from typing import List, Dict, Any, Optional
import aiohttp
from ..config.config import (
    OPENAI_API_KEY, JINA_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSION,
    GPT_MODEL_INITIAL, GPT_MODEL_BEST_MATCH, GPT_MODEL_LEVEL1
)
from ..utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

async def get_embedding(text: str, semaphore: asyncio.Semaphore, rate_limiter: RateLimiter, task_type: str = 'classification'):
    """
    Get embeddings using Jina.ai API with improved retry logic.
    
    Args:
        text (str): Text to embed
        semaphore (asyncio.Semaphore): Semaphore for controlling concurrent requests
        rate_limiter (RateLimiter): Rate limiter to avoid hitting API limits
        task_type (str): Type of embedding task (classification, search, etc.)
        
    Returns:
        List[float]: Embedding vector or None if failed
    """
    logger.debug(f"get_embedding called with text: {text[:50]}..." if len(text) > 50 else f"get_embedding called with text: {text}")
    max_retries = 8
    url = 'https://api.jina.ai/v1/embeddings'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {JINA_API_KEY}'
    }
    
    # Prepare the text
    text = str(text).strip()
    if not text:
        text = "unknown"
    
    for attempt in range(max_retries):
        try:
            # Add jitter to backoff to avoid thundering herd
            jitter = random.uniform(0.1, 0.5)
            backoff_time = (2 ** attempt) + jitter
            
            logger.debug(f"Attempt {attempt+1}/{max_retries} to get embedding.")
            await rate_limiter.acquire()
            
            async with semaphore:
                data = {
                    "model": EMBEDDING_MODEL,
                    "task": task_type,
                    "late_chunking": False,
                    "dimensions": EMBEDDING_DIMENSION,
                    "embedding_type": "float",
                    "input": [text]
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=data, timeout=30) as response:
                        if response.status == 200:
                            result = await response.json()
                            embedding = result['data'][0]['embedding']
                            
                            if len(embedding) != EMBEDDING_DIMENSION:
                                raise ValueError(f"Expected embedding dimension {EMBEDDING_DIMENSION}, got {len(embedding)}")

                            logger.debug("Embedding dimension verified.")
                            return embedding

                        else:
                            error_text = await response.text()
                            error_message = f"Error from Jina.ai API (status {response.status}): {error_text[:200]}..."
                            
                            # Detailed logging for various error codes
                            if response.status == 502:
                                logger.warning(f"Encountered Jina.ai 502 Bad Gateway error on attempt {attempt+1}/{max_retries}. Retrying in {backoff_time:.2f}s...")
                            elif response.status in [429, 500, 503, 504]:
                                logger.warning(f"Jina.ai returned status {response.status} on attempt {attempt+1}/{max_retries}. Retrying in {backoff_time:.2f}s...")
                            else:
                                logger.warning(error_message)
                                
                            if response.status in [429, 500, 502, 503, 504]:
                                await asyncio.sleep(backoff_time)
                                continue
                            return None

        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting embedding (attempt {attempt+1}/{max_retries}). Retrying in {backoff_time:.2f}s...")
            await asyncio.sleep(backoff_time)
            continue
        except Exception as e:
            logger.warning(f"Error getting embedding (attempt {attempt+1}/{max_retries}): {e}. Retrying in {backoff_time:.2f}s...")
            if attempt < max_retries - 1:
                await asyncio.sleep(backoff_time)
            else:
                logger.error(f"Failed to get embedding after {max_retries} attempts: {e}")
                return None

    # If we reach here, all retries have failed
    logger.error(f"All {max_retries} attempts to get embedding failed")
    return None

def categorize_supplier_type(supplier_name: str) -> str:
    """
    Categorize supplier by type to assist with classification.
    
    Args:
        supplier_name: Name of the supplier
        
    Returns:
        str: Category of the supplier (AIRLINE, LOGISTICS, HOTEL, FINANCIAL, or OTHER)
    """
    supplier_lower = supplier_name.lower()
    
    # Airlines (T&E - Passenger Travel)
    airlines = ["delta", "united", "american airlines", "southwest", "jetblue", "lufthansa", 
                "british airways", "air france", "emirates", "qantas", "air canada"]
    
    # Logistics/Freight (Transportation of Goods)
    logistics = ["ups freight", "fedex freight", "dhl", "xpo logistics", "c.h. robinson", 
                 "expeditors", "kuehne", "maersk", "db schenker", "ceva", "freight"]
    
    # Hotels (T&E - Lodging)
    hotels = ["marriott", "hilton", "hyatt", "sheraton", "westin", "holiday inn", 
              "intercontinental", "radisson", "hotel", "motel", "resort"]
    
    # Financial Institutions (Potential T&E via cards)
    financial = ["american express", "amex", "visa", "mastercard", "bank of america", 
                 "chase", "citi", "wells fargo", "capital one", "bank", "card services"]
    
    # Check supplier type
    for airline in airlines:
        if airline in supplier_lower:
            return "AIRLINE"
    
    for provider in logistics:
        if provider in supplier_lower:
            return "LOGISTICS"
    
    for hotel in hotels:
        if hotel in supplier_lower:
            return "HOTEL"
    
    for institution in financial:
        if institution in supplier_lower:
            return "FINANCIAL"
    
    return "OTHER"

async def classify_transaction_initial(
    session, 
    supplier_name: str, 
    primary_descriptor: str,
    customer_industry_description: str, 
    semaphore: asyncio.Semaphore, 
    rate_limiter: RateLimiter,
    alternate_descriptor_1: str = None,
    alternate_descriptor_2: str = None,
    transaction_value: float = 0.0
) -> Optional[Dict[str, Any]]:
    """
    Initial classification of a transaction using OpenAI API.
    
    Args:
        session: aiohttp session
        supplier_name: Name of the supplier
        primary_descriptor: Primary description of the transaction
        alternate_descriptor_1: First alternate description (optional)
        alternate_descriptor_2: Second alternate description (optional)
        customer_industry_description: Description of the customer's industry
        transaction_value: Value of the transaction (optional)
        semaphore: Semaphore for controlling concurrent requests
        rate_limiter: Rate limiter to avoid hitting API limits
        
    Returns:
        Optional[Dict[str, Any]]: Classification result or None if failed
    """
    logger.debug("classify_transaction_initial called.")
    max_retries = 5

    system_prompt = """
You are an expert in spend classification. You will be given context about a customer and their industry, a supplier name, and transaction descriptors.
Use all this context to produce a more accurate classification.

Your tasks:
1. Provide a 'cleansed_description':
   - Retain part numbers, SKUs, and product codes ONLY WHERE THEY HELP IDENTIFY THE PRODUCT
   - Remove numbers or random codes which do not help in identification of the spend category
   - Remove transaction identifiers (invoice/PO/transaction IDs)
   - Highlight general product categories or service types
   - Standardize currency to "USD"
   - Consider the supplier, what they might provide, and the customer's industry
   - Use supplier name and customer industry description to infer product/service type
   - Always provide a description, even if generalized
   - Always provide the description in English
   - Avoid mentioning the purchasing customers specific name in the description for privacy reasons
   
   SPECIAL CASES TO HANDLE CAREFULLY:
   - For TRANSPORTATION: When the transaction involves freight/logistics, include the mode of transportation (truck, rail, ocean, air, parcel) and any special handling requirements when this information is available
   - For AIRLINES: Classify as passenger travel (T&E), not freight, unless explicitly stated otherwise
   - For FINANCIAL INSTITUTIONS: When descriptors suggest travel or expenses, focus on the actual expense type rather than financial services

2. 'sourceable_flag':
   - TRUE if item/service is clear and has enough detail for a competitive sourcing event.
   - FALSE if complex, unique, or specialized. Default FALSE if uncertain.

3. 'single_word':
   - Produce a single English word that best describes or labels the transaction.
   - For example, if it's about office supplies, you might say "office".
   - If it's about consulting services, you might say "consulting".
   - Return all lowercase, no punctuation.
   - If uncertain, return the category_description word.

4. 'transaction_type':
   - Choose one from the following options: "Indirect Goods", "Indirect Services", "Direct Materials", "Direct Services", "Capital Equipment".

5. 'searchable':
   - Return TRUE ONLY if you believe that performing a web search using the supplier name and transaction description (including any supplier part numbers) would greatly enhance the accuracy of classification; otherwise, return FALSE.

6. rfp_description:
   - Provide a clear, concise description of the product or service for use in a Request for Quote.
   - Include only the product/service name, key specifications (such as dimensions, capacity, packaging), and any necessary part numbers or codes that clearly identify the item.
   - Do not include any pricing instructions, vendor requests, or additional commentary.
   - Use standard industry terminology and avoid any preamble or non-essential language.

IMPORTANT: Your response must be valid JSON. Double-check all quotes, brackets, and special characters.
"""

    # Categorize supplier type to provide better guidance
    supplier_type = categorize_supplier_type(supplier_name)
    
    user_message = f"""
Customer Industry Description: {customer_industry_description}
Supplier: {supplier_name}
Primary Descriptor: {primary_descriptor}
Alternate Descriptor 1: {alternate_descriptor_1 if alternate_descriptor_1 else "Not provided"}
Alternate Descriptor 2: {alternate_descriptor_2 if alternate_descriptor_2 else "Not provided"}
Transaction Value: ${transaction_value:,.2f}

IMPORTANT CONTEXT:
"""

    # Add specific guidance based on supplier type
    if supplier_type == "AIRLINE":
        user_message += "- This supplier appears to be an AIRLINE - focus on PASSENGER TRAVEL aspects, not freight\n"
    elif supplier_type == "LOGISTICS":
        user_message += "- This supplier appears to be a LOGISTICS provider - try to identify the specific mode of transportation and any special handling requirements\n"
    elif supplier_type == "HOTEL":
        user_message += "- This supplier appears to be a HOTEL - focus on LODGING aspects of the transaction\n"
    elif supplier_type == "FINANCIAL":
        user_message += "- This supplier appears to be a FINANCIAL INSTITUTION - if the descriptors suggest travel or expenses, focus on the actual expense type\n"

    json_schema = {
        "name": "initial_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "cleansed_description": {"type": "string"},
                "sourceable_flag": {"type": "boolean"},
                "single_word": {"type": "string"},
                "transaction_type": {"type": "string", "enum": ["Indirect Goods", "Indirect Services", "Direct Materials", "Direct Services", "Capital Equipment"]},
                "searchable": {"type": "boolean"},
                "rfp_description": {"type": "string"}
            },
            "required": ["cleansed_description", "sourceable_flag", "single_word", "transaction_type", "searchable", "rfp_description"],
            "additionalProperties": False
        }
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    await rate_limiter.acquire()
                    logger.debug(f"Making LLM request attempt {attempt + 1}/{max_retries}")

                    logger.info(
                        f"Sending LLM request (classify_transaction_initial) - "
                        f"model={GPT_MODEL_INITIAL}, messages length={len(messages)}"
                    )

                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                        json={
                            "model": GPT_MODEL_INITIAL,
                            "messages": messages,
                            "max_tokens": 4096,
                            "temperature": 0.1,
                            "response_format": {
                                "type": "json_schema",
                                "json_schema": json_schema
                            }
                        }
                    ) as response:
                        if response.status == 200:
                            logger.debug("Initial classification response received successfully.")
                            llm_choice = await response.json()
                            llm_response = llm_choice['choices'][0]['message']['content']
                            
                            try:
                                llm_output = json.loads(llm_response)
                                logger.debug("Successfully parsed initial classification JSON response.")
                                return llm_output
                                
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON parsing failed on attempt {attempt + 1}/{max_retries}: {e}")
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(2 ** attempt)
                                    continue
                                else:
                                    logger.error(f"Failed to get valid JSON after {max_retries} attempts: {e}")
                                    return None
                        else:
                            error_message = await response.text()
                            logger.warning(f"Error in initial classification call (status {response.status}): {error_message}")
                            if response.status in [429, 500, 502, 503, 504]:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return None

                except Exception as e:
                    logger.warning(f"Error in LLM request (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        logger.error(f"Failed to complete LLM request after {max_retries} attempts: {e}")
                        return None

    except Exception as e:
        logger.error(f"Unexpected error in classify_transaction_initial: {e}")
        return None

    return None

async def select_best_match(
    session, 
    transaction_id: str, 
    transaction_description: str, 
    supplier_name: str, 
    llm_description: str, 
    transaction_value: float, 
    l3_matches: List[Dict[str, Any]], 
    l4_matches: List[Dict[str, Any]], 
    customer_industry_description: str, 
    semaphore: asyncio.Semaphore, 
    rate_limiter: RateLimiter
) -> Optional[Dict[str, Any]]:
    """
    Select the best match from L3 and L4 categories using OpenAI API.
    
    Args:
        session: aiohttp session
        transaction_id: ID of the transaction
        transaction_description: Description of the transaction
        supplier_name: Name of the supplier
        llm_description: Cleansed description from initial classification
        transaction_value: Value of the transaction
        l3_matches: List of L3 category matches
        l4_matches: List of L4 category matches
        customer_industry_description: Description of the customer's industry
        semaphore: Semaphore for controlling concurrent requests
        rate_limiter: Rate limiter to avoid hitting API limits
        
    Returns:
        Optional[Dict[str, Any]]: Best match or None if failed
    """
    logger.info(f"Passing {len(l3_matches) + len(l4_matches)} total matches to LLM for transaction {transaction_id}")
    logger.debug("select_best_match called.")

    max_retries = 5
    
    valid_category_ids = [match['UNSPSC_ID'] for match in l3_matches + l4_matches]
    system_prompt = """
You are an expert in spend classification. Your role is to classify spend and invoice transactions for enterprise customers purchasing diverse goods and services.

EVALUATION FRAMEWORK:
1. SEMANTIC RELEVANCE (70% weight)
    - Analyze core business purpose alignment
    - Consider industry context and typical categorization
    - Evaluate functional similarity between transaction description and category description
    - Match domain-specific terminology
    - Incorporate both explicit descriptions and implicit business context for transactions and category descriptions

2. SUPPLIER ANALYSIS (15% weight)
    - Assess the primary business classification of the supplier
    - Evaluate typical industry categorization patterns
    - Consider the supplier's main product/service offerings if well known

3. CONTEXTUAL SIGNALS (15% weight)
    - Examine transaction value alignment with category norms
    - Consider customer industry relevance
    - Evaluate overall semantic relevance of the transaction description

DECISION RULES:
    - You MUST select from the provided category IDs only.

    - When choosing between L3 and L4:
        * Select L4 only when the transaction clearly fits a specialized subcategory and is directly appropriate for the customer industry
        * Default to L3 in cases of ambiguity or mixed signals
        * Always consider the customer industry context for granularity decisions

- Confidence Scoring Requirements:
    * 9-10: Perfect alignment across all evaluation criteria
    * 8: Strong alignment with minor uncertainty
    * 7: Strong category fit with some ambiguity
    * 6: Some category fit with serious ambiguity
    * 5: Clear category fit with some ambiguity
    * 3-4: Best available match from choices provided with some uncertainty
    * 1-2: Forced match with major uncertainty

YOUR TASK:
1. For each potential match from the provided UNSPSC_IDs, perform a step-by-step evaluation:
    a. Score Semantic Relevance based on the core business purpose, industry context, functional similarity, and domain terminology.
    b. Evaluate Supplier Analysis by considering the supplier's main business and typical categorization patterns.
    c. Analyze Contextual Signals including transaction value and customer industry details.
    d. Document your evaluation for each candidate internally (chain-of-thought) but DO NOT include these internal details in your final output.
   
2. After scoring all candidates, select the highest scoring match from the provided category IDs.

3. Provide a structured justification that references each part of the evaluation framework (e.g., bullet points or a summary table) and explains why the selected code is the best fit.

4. Assign a confidence score following the strict scoring guide. Your explanation should clearly link the evaluation evidence to the chosen score.

5. ***CRITICAL*** You MUST ONLY select a classification_code that appears in the provided UNSPSC_ID list.

6. FINAL OUTPUT REQUIREMENTS:
   - Your response must be valid JSON.

IMPORTANT: Double-check all quotes, brackets, and special characters in your final JSON output. Do not reveal your internal chain-of-thought.
"""

    system_prompt += f"\n\nValid UNSPSC_IDs you can select from: {', '.join(valid_category_ids)}"

    user_message = f"""
Customer Industry Description: {customer_industry_description}
Transaction details:
- Transaction ID: {transaction_id}
- Supplier Name: {supplier_name}
- Transaction Description: {transaction_description}
- LLM Description: {llm_description}
- Transaction Value: {transaction_value}

Potential Matches:
"""
    all_matches = l3_matches + l4_matches
    for match in all_matches:
        user_message += f"- ID: {match['UNSPSC_ID']} - {match['Description']} (Level: {match['level']}, Score: {match.get('Score', 'N/A')})\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    json_schema = {
        "name": "financial_transaction_categorization",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "The unique identifier for the transaction."
                },
                "supplier_name": {
                    "type": "string",
                    "description": "The name of the supplier."
                },
                "category_classification": {
                    "type": "string",
                    "description": "The selected category for the spend or invoice."
                },
                "classification_code": {
                    "type": "string",
                    "enum": valid_category_ids,
                    "description": "The classification code which best matches the original transaction selected from the valid options only."
                },
                "level": {
                    "type": "string",
                    "description": "Level of the selected category (L3 or L4).",
                    "enum": ["L3", "L4"]
                },
                "reason": {
                    "type": "string",
                    "description": "A concise summary of your reasoning for your choice and your confidence score."
                },
                "confidence_score": {
                    "type": "number",
                    "description": "An integer between 0 and 10 for confidence."
                }
            },
            "required": [
                "transaction_id",
                "supplier_name",
                "category_classification",
                "classification_code",
                "level",
                "reason",
                "confidence_score"
            ],
            "additionalProperties": False
        }
    }

    try:
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    await rate_limiter.acquire()
                    logger.debug(f"Making LLM request attempt {attempt + 1}/{max_retries}")

                    logger.info(
                        f"Sending LLM request (select_best_match) - "
                        f"model={GPT_MODEL_BEST_MATCH}, messages length={len(messages)}"
                    )
                    
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                        json={
                            "model": GPT_MODEL_BEST_MATCH,
                            "messages": messages,
                            "max_tokens": 2048,
                            "temperature": 0.1,
                            "response_format": {
                                "type": "json_schema",
                                "json_schema": json_schema
                            }
                        }
                    ) as response:
                        if response.status == 200:
                            llm_choice = await response.json()
                            llm_response = llm_choice['choices'][0]['message']['content']
                            
                            try:
                                llm_output = json.loads(llm_response)
                                
                                classification_code = llm_output.get("classification_code", "")
                                if classification_code not in valid_category_ids:
                                    logger.error(f"LLM selected invalid category ID {classification_code} for transaction {transaction_id}")
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(2 ** attempt)
                                        continue
                                    return None

                                chosen_match = next((m for m in all_matches if m['UNSPSC_ID'] == classification_code), None)
                                if chosen_match:
                                    category_description = chosen_match['Description']
                                else:
                                    category_description = llm_output.get("category_classification", "")

                                return {
                                    "Transaction_ID": transaction_id,
                                    "Supplier_Name": supplier_name,
                                    "Transaction_Description": transaction_description,
                                    "LLM_Description": llm_description,
                                    "Transaction_Value": transaction_value,
                                    "Category_ID": classification_code,
                                    "Category_Description": category_description,
                                    "level": llm_output.get("level", ""),
                                    "Reason": llm_output.get("reason", ""),
                                    "Confidence_Score": llm_output.get("confidence_score", ""),
                                    "Match_Score": chosen_match.get('Score', 0) if chosen_match else 0
                                }
                                
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON parsing failed on attempt {attempt + 1}/{max_retries} for transaction {transaction_id}: {e}")
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(2 ** attempt)
                                    continue
                                else:
                                    logger.error(f"Failed to get valid JSON after {max_retries} attempts for transaction {transaction_id}: {e}")
                                    return None
                        else:
                            error_message = await response.text()
                            logger.warning(f"Error selecting best match (status code {response.status}): {error_message}")
                            if response.status in [429, 500, 502, 503, 504]:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return None

                except Exception as e:
                    logger.warning(f"Error in LLM request (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        logger.error(f"Failed to complete LLM request after {max_retries} attempts: {e}")
                        return None

    except Exception as e:
        logger.error(f"Unexpected error in select_best_match for transaction {transaction_id}: {e}")
        return None

    return None

async def get_level1_categories(
    session, 
    supplier_name: str, 
    transaction_description: str, 
    llm_description: str, 
    customer_industry_description: str, 
    transaction_value: float,
    industry_filter: int,
    semaphore: asyncio.Semaphore, 
    rate_limiter: RateLimiter
) -> Optional[List[int]]:
    """
    Get level 1 categories for a transaction using OpenAI API.
    
    Args:
        session: aiohttp session
        supplier_name: Name of the supplier
        transaction_description: Description of the transaction
        llm_description: Cleansed description from initial classification
        customer_industry_description: Description of the customer's industry
        transaction_value: Value of the transaction
        industry_filter: Industry filter value
        semaphore: Semaphore for controlling concurrent requests
        rate_limiter: Rate limiter to avoid hitting API limits
        
    Returns:
        Optional[List[int]]: List of level 1 category IDs or None if failed
    """
    logger.debug("get_level1_categories called.")
    max_retries = 5

    # Import the database utility to fetch relevant UNSPSC sectors
    from .database import fetch_relevant_unspsc_sectors
    
    # Get relevant UNSPSC sectors for the industry filter
    relevant_sectors = await fetch_relevant_unspsc_sectors(industry_filter)
    if not relevant_sectors:
        logger.warning(f"No matching UNSPSC sectors found for industry_filter={industry_filter}.")
        return []

    # Build the available categories string
    available_categories_str = ""
    for sector in relevant_sectors:
        sector_num = sector["unspsc_sector"]
        sector_desc = sector["unspsc_sector_description"]
        available_categories_str += f"{sector_num}. {sector_desc}\n"

    # Get the valid sectors list for validation
    valid_sectors = [sector["unspsc_sector"] for sector in relevant_sectors]

    system_prompt = f"""
You are an expert in spend classification and procurement analytics, specializing in UNSPSC taxonomy. Your task is to analyze transaction details and select the most relevant Level 1 UNSPSC categories based on the provided information.

EVALUATION FRAMEWORK:

1. Primary Business Purpose (Highest Priority)
   - What specific product or service is being purchased?
   - What is the direct business application?
   - How does the transaction value align with typical category benchmarks?

2. Supplier Profile
   - What is the supplier's primary business classification?
   - What are their core product/service offerings?
   - Are they a specialized or general vendor?

3. Industry Context
   - What are standard procurement patterns in this industry?
   - How does this transaction align with typical industry spending?
   - Are there regulatory or compliance considerations?

SELECTION RULES:
- Select exactly 3 categories, ranked by relevance
- Categories must be distinct with no overlapping primary functions
- Prioritize specific categories over general ones when evidence supports it

IMPORTANT: Your response must be valid JSON. Double-check all quotes, brackets, and special characters.

Your response must be a JSON object with a single field "category_numbers" containing an array of exactly 3 integers. 
Example:
{{
    "category_numbers": [1,22,45]
}}
"""

    user_message = f"""
INPUT ANALYSIS:
1. Transaction Profile
   - Supplier Name: {supplier_name}
   - Transaction Description: {transaction_description}
   - Normalized Description: {llm_description}
   - Transaction Value: ${transaction_value:,.2f}

2. Business Context
   - Customer Industry: {customer_industry_description}

AVAILABLE CATEGORIES:
{available_categories_str}

TASK:
1. Analyze the transaction value and description
2. Consider the supplier's typical business activities
3. Account for the customer's industry context
4. Only select categories from the above AVAILABLE CATEGORIES list that represent goods/services the customer would likely purchase
5. Select exactly 3 unspsc_sector numbers (the 'unspsc_sector' integers) that best match
"""

    json_schema = {
        "name": "level_1_category_selection",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "category_numbers": {
                    "type": "array",
                    "items": {
                        "type": "number",
                        "enum": valid_sectors,  # Only allow numbers from the valid_sectors list
                        "description": "An array of 3 integers representing the most likely categories."
                    }
                }
            },
            "required": ["category_numbers"],
            "additionalProperties": False
        }
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    await rate_limiter.acquire()
                    logger.debug(f"Making LLM request attempt {attempt + 1}/{max_retries}")

                    logger.info(
                        f"Sending LLM request (get_level1_categories) - "
                        f"model={GPT_MODEL_LEVEL1}, messages length={len(messages)}"
                    )

                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                        json={
                            "model": GPT_MODEL_LEVEL1,
                            "messages": messages,
                            "max_tokens": 100,
                            "temperature": 0.1,
                            "response_format": {
                                "type": "json_schema",
                                "json_schema": json_schema
                            }
                        }
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            response_content = result['choices'][0]['message']['content']
                            
                            try:
                                parsed_response = json.loads(response_content)
                                if 'category_numbers' in parsed_response:
                                    category_numbers = parsed_response['category_numbers']

                                    # Validate that all returned codes are in valid_sectors
                                    if all(code in valid_sectors for code in category_numbers):
                                        logger.debug("Successfully retrieved and validated level 1 category numbers")
                                        return category_numbers
                                    else:
                                        logger.warning(
                                            f"LLM returned invalid sector codes: {category_numbers}. "
                                            f"Valid sectors = {valid_sectors}. Retrying..."
                                        )
                                        if attempt < max_retries - 1:
                                            # Wait a bit and then retry the LLM
                                            await asyncio.sleep(2 ** attempt)
                                            continue
                                        else:
                                            logger.error("Failed to get valid sector codes after all attempts.")
                                            return None
                                else:
                                    logger.warning(f"Invalid response format: {parsed_response}")
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(2 ** attempt)
                                        continue
                                    else:
                                        logger.error("Failed to get valid category numbers after all attempts.")
                                        return None
                                    
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON parsing failed on attempt {attempt + 1}/{max_retries}: {e}")
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(2 ** attempt)
                                    continue
                                else:
                                    logger.error(f"Failed to get valid JSON after {max_retries} attempts: {e}")
                                    return None
                        else:
                            error_message = await response.text()
                            logger.warning(f"Error getting level 1 categories (status code {response.status}): {error_message}")
                            if response.status in [429, 500, 502, 503, 504]:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return None

                except Exception as e:
                    logger.warning(f"Error in LLM request (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        logger.error(f"Failed to complete LLM request after {max_retries} attempts: {e}")
                        return None

    except Exception as e:
        logger.error(f"Unexpected error in get_level1_categories: {e}")
        return None

    return None
