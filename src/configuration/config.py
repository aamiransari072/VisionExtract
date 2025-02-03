import os



class Configuration:
    def __init__(self):
        pass


    def get_prompt(self):
        prompt = f"""
                 You are an AI specialized in extracting structured data from invoices. 
                Your task is to extract key details from the given text and return ONLY a JSON output.

                Extract the following fields:
                - **invoice_number**: The invoice number.
                - **date**: The invoice date.
                - **vendor**: The vendor name.
                - **total_amount**: The total invoice amount.
                - **items**: A list of items, where each item has:
                - **name**: The name of the item.
                - **quantity**: The quantity of the item.
                - **price**: The price of the item.

                Output Format (JSON)
                Return ONLY a valid JSON object without any additional text no preamble no markdown only Valid JSON.


                
                {{
                "invoice_number": "INV-12345",
                "date": "2024-02-02",
                "vendor": "ABC Corp",
                "total_amount": "$500",
                "items": [
                    {{"name": "Laptop", "quantity": 1, "price": "$1200"}},
                    {{"name": "Mouse", "quantity": 2, "price": "$50"}}
                ]
                }}

                """
        return prompt