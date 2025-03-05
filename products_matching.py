import os
import json
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain.chat_models import OpenAI  # Corrected import for LangChain

# Configure logging
logging.basicConfig(filename="logs.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

# Initialize OpenAI client
client = OpenAI(api_key="your-api-key")

class FileEventHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(".json"):
            logging.info(f"File change detected: {event.src_path}")
            process_files()

def load_data(product_file, guideline_file):
    if not os.path.exists(product_file) or not os.path.exists(guideline_file):
        logging.error("One or both input files are missing.")
        return None, None
    
    with open(product_file, 'r', encoding='utf-8') as f:
        products = json.load(f)
    with open(guideline_file, 'r', encoding='utf-8') as f:
        guidelines = json.load(f)
    return products, guidelines

def query_llm(product, guideline):
    """Use LLM to determine if a product matches the guideline."""
    prompt = f"""
    Product: {json.dumps(product)}
    Guideline: {json.dumps(guideline)}
    Does this product match the guideline? Provide a yes/no answer and explain why.
    """
    response = client.predict(prompt)
    return response.strip()

def match_products(products, guidelines):
    if products is None or guidelines is None:
        return [], []
    
    matched, unmatched = [], []
    
    for product in products:
        category = product.get('Category')
        guideline = next((g for g in guidelines if g.get('Category') == category), None)
        
        if not guideline:
            unmatched.append((product, 'Category not found in guidelines'))
            continue
        
        llm_response = query_llm(product, guideline)
        
        if "yes" in llm_response.lower():
            matched.append(product)
        else:
            unmatched.append((product, llm_response))
    
    return matched, unmatched

def save_results(matched, unmatched, output_dir='output'):
    os.makedirs(f"{output_dir}/Matching", exist_ok=True)
    os.makedirs(f"{output_dir}/Unmatching", exist_ok=True)
    
    with open(f"{output_dir}/Matching/matched_products.json", 'w') as f:
        json.dump(matched, f, indent=4)
    with open(f"{output_dir}/Unmatching/unmatched_products.json", 'w') as f:
        json.dump([u[0] for u in unmatched], f, indent=4)
    
    with open(f"{output_dir}/Unmatching/unmatched_reasons.txt", 'w') as f:
        for item in unmatched:
            f.write(f"{item[0]['Name']}: {item[1]}\n")
    
    logging.info("Results saved successfully.")

def process_files():
    try:
        product_file = "orders/products.json"
        guideline_file = "guidelines/guidelines.json"
        
        products, guidelines = load_data(product_file, guideline_file)
        if products is None or guidelines is None:
            logging.error("Processing aborted due to missing files.")
            return
        
        matched, unmatched = match_products(products, guidelines)
        save_results(matched, unmatched)
        logging.info("Processing completed!")
    except Exception as e:
        logging.error(f"Error processing files: {e}")

def main():
    logging.info("Bot started, listening for file changes...")
    event_handler = FileEventHandler()
    observer = Observer()
    observer.schedule(event_handler, "orders", recursive=False)
    observer.schedule(event_handler, "guidelines", recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    logging.info("Bot stopped.")
    
if __name__ == "__main__":
    main()
