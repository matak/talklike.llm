#!/usr/bin/env python3
"""
Training Data Moderator Script with OpenAI Moderation

This script:
1. Uses OpenAI's moderation API to check content
2. Converts JSON format to JSONL format (one JSON object per line)
3. Ensures each conversation ends with an assistant message
4. Filters out flagged content
"""

import os
import json
import sys
from typing import List, Dict, Any
import logging
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('moderation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_content(content: str) -> Dict[str, Any]:
    """
    Check content using OpenAI's moderation API.
    
    Args:
        content: Content to moderate
        
    Returns:
        Dict with moderation results
    """
    # Maximum character length for each content segment
    max_length = 1500
    parts = [content[i : i + max_length] for i in range(0, len(content), max_length)]

    moderation_results = []
    flagged = False
    categories = set()

    # Moderation check for each part
    for part in parts:
        try:
            response = client.moderations.create(input=part)
            result = response.results[0]
            moderation_results.append(result)
            if result.flagged:
                flagged = True
                categories.update(result.categories)
        except Exception as e:
            logger.error(f"Error during moderation of content segment: {e}")

    # Combine results
    if flagged:
        logger.warning("Some content segments were flagged")
        return {"flagged": True, "categories": list(categories)}
    else:
        return {"flagged": False, "categories": []}

def validate_conversation(messages: List[Dict[str, Any]]) -> bool:
    """
    Validate a conversation structure.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        bool: True if conversation is valid, False otherwise
    """
    if not messages:
        return False
    
    # Check if conversation ends with assistant message
    if messages[-1]["role"] != "assistant":
        logger.warning(f"Conversation does not end with assistant message. Last role: {messages[-1]['role']}")
        return False
    
    # Check for valid roles
    valid_roles = {"system", "user", "assistant"}
    for msg in messages:
        if msg["role"] not in valid_roles:
            logger.warning(f"Invalid role found: {msg['role']}")
            return False
        
        if "content" not in msg or not msg["content"]:
            logger.warning(f"Message missing content: {msg}")
            return False
    
    return True

def split_conversations(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split a large conversation into individual conversation pairs.
    
    Args:
        data: The original data structure
        
    Returns:
        List of individual conversation objects
    """
    messages = data.get("messages", [])
    if not messages:
        return []
    
    conversations = []
    current_conversation = []
    
    for i, message in enumerate(messages):
        if message["role"] == "system":
            # Start a new conversation with system message
            if current_conversation:
                conversations.append({"messages": current_conversation})
            current_conversation = [message]
        elif message["role"] in ["user", "assistant"]:
            current_conversation.append(message)
            
            # If this is an assistant message, check if we should end the conversation
            if message["role"] == "assistant":
                # Look ahead to see if next message is user (continuing conversation)
                # or system (new conversation)
                if i + 1 < len(messages):
                    next_message = messages[i + 1]
                    if next_message["role"] == "system":
                        # End current conversation
                        conversations.append({"messages": current_conversation})
                        current_conversation = []
                else:
                    # Last message, end conversation
                    conversations.append({"messages": current_conversation})
                    current_conversation = []
    
    # Add any remaining conversation
    if current_conversation:
        conversations.append({"messages": current_conversation})
    
    return conversations

def moderate_data(input_file: str, output_file: str) -> None:
    """
    Moderate and fix the training data.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSONL file
    """
    logger.info(f"Starting moderation of {input_file}")
    
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info("Successfully loaded input data")
        
        # Split into individual conversations
        conversations = split_conversations(data)
        logger.info(f"Split into {len(conversations)} conversations")
        
        # Validate and filter conversations
        valid_conversations = []
        invalid_count = 0
        flagged_count = 0
        
        for i, conv in enumerate(conversations):
            # First validate structure
            if not validate_conversation(conv["messages"]):
                invalid_count += 1
                logger.warning(f"Invalid conversation {i+1}: {conv}")
                continue
            
            # Then check content moderation
            content = " ".join([msg["content"] for msg in conv["messages"] if "content" in msg])
            moderation_response = check_content(content)
            
            if moderation_response["flagged"]:
                flagged_count += 1
                categories = moderation_response["categories"]
                logger.warning(f"Conversation {i+1} flagged for categories: {categories}")
                continue
            
            valid_conversations.append(conv)
        
        logger.info(f"Valid conversations: {len(valid_conversations)}")
        logger.info(f"Invalid conversations: {invalid_count}")
        logger.info(f"Flagged conversations: {flagged_count}")
        
        # Write to JSONL format
        with open(output_file, 'w', encoding='utf-8') as f:
            for conv in valid_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
        
        logger.info(f"Successfully wrote {len(valid_conversations)} conversations to {output_file}")
        
        # Create a summary report
        create_summary_report(valid_conversations, invalid_count, flagged_count, output_file)
        
    except Exception as e:
        logger.error(f"Error during moderation: {e}")
        raise

def create_summary_report(conversations: List[Dict[str, Any]], invalid_count: int, flagged_count: int, output_file: str) -> None:
    """
    Create a summary report of the moderation process.
    
    Args:
        conversations: List of valid conversations
        invalid_count: Number of invalid conversations
        flagged_count: Number of flagged conversations
        output_file: Path to output file
    """
    report_file = output_file.replace('.jsonl', '_report.txt')
    total_processed = len(conversations) + invalid_count + flagged_count
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("TRAINING DATA MODERATION REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total conversations processed: {total_processed}\n")
        f.write(f"Valid conversations: {len(conversations)}\n")
        f.write(f"Invalid conversations: {invalid_count}\n")
        f.write(f"Flagged conversations: {flagged_count}\n")
        f.write(f"Success rate: {len(conversations)/total_processed*100:.1f}%\n\n")
        
        f.write("CONVERSATION STATISTICS:\n")
        f.write("-" * 25 + "\n")
        
        if conversations:
            total_messages = sum(len(conv["messages"]) for conv in conversations)
            f.write(f"Total messages: {total_messages}\n")
            f.write(f"Average messages per conversation: {total_messages/len(conversations):.1f}\n")
            
            # Count message types
            role_counts = {}
            for conv in conversations:
                for msg in conv["messages"]:
                    role = msg["role"]
                    role_counts[role] = role_counts.get(role, 0) + 1
            
            f.write("\nMessage type distribution:\n")
            for role, count in role_counts.items():
                f.write(f"  {role}: {count}\n")
        else:
            f.write("No valid conversations to analyze.\n")
    
    logger.info(f"Summary report written to {report_file}")

def main():
    """Main function."""
    input_file = "data/all.jsonl"
    output_file = "data/moderated_training_data.jsonl"
    
    try:
        moderate_data(input_file, output_file)
        logger.info("Moderation completed successfully!")
        
        # Also create a backup of the original file
        import shutil
        backup_file = "data/all_backup.jsonl"
        shutil.copy2(input_file, backup_file)
        logger.info(f"Original file backed up to {backup_file}")
        
    except Exception as e:
        logger.error(f"Moderation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 