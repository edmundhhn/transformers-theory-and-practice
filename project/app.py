import torch
from transformers import BertTokenizer
from regression_models import BERTRegression
import gradio as gr

max_len = 80

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load model architecture
bertregressor = BERTRegression()
bertregressor.load_state_dict(torch.load('bert_regression_model.pth', map_location=torch.device('cpu')))
bertregressor.eval()

def predict_price(name, item_condition, category, brand_name, shipping_included, item_description):
    print((name, item_condition, category, brand_name, shipping_included, item_description))
    # Preprocess Input
    if shipping_included:
        shipping_str = "Includes Shipping"
    else:
        shipping_str = "No Shipping"
        
    combined = "Item Name: " + name + \
            " Description: " + item_description + \
            " Condition: " + item_condition + \
            " Category: " + category + \
            " Brand " + brand_name + \
            " Shipping: " + shipping_str
    
    inputs = tokenizer.encode_plus(
        combined,
        None,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        output = bertregressor(input_ids, attention_mask)
    
    return output.item()

    
demo = gr.Interface(
    
    fn = predict_price,
    
    inputs = [gr.Textbox(label="Item Name"), 
              gr.Dropdown(['Poor', 'Okay', 'Good', 'Excellent', 'Like New'], label="Item Condition", info="What condition is the item in?"),
              gr.Textbox(label="Category on Mercari"),
              gr.Textbox(label="Brand"),
              gr.Checkbox(label="Shipping Included"),
              gr.Textbox(label="Description")
             ],
    
    #outputs = gr.Textbox()
    outputs= gr.Number()
)


demo.launch()