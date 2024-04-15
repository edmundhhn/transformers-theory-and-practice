# Automated Pricing for Sellers on Mercari
__Authors: Thao Nguyen, Edmund Hui__

Play with our project on [Huggingface Spaces](https://huggingface.co/spaces/edmundhui/mercari-price-prediction) ! We take 25% comission for every successful sale using our application.

## Overview

- __Context__: Mercari is a Japanese e-comemrce company. Their main product, the Mercari marketplace app is a place for buyers and sellers to buy sell and trade items securely and hosts over 23 million users every year. 

- __Problem__: A major obstacle for the sellers on the Mercari marketplace is the difficulty of setting a competitive price for their product. Accurately setting product prices is paramount for maximizing profits and staying ahead of the competition, whilst incorrectly setting prices could lead to revenue loss through selling too cheap, or delayed sales/no sale by selling too expensive.

- __Approach__: To assist the seller in their price determination, we fine tune a BERT model on a large amount of user inputted descriptions of [products which sold on the Mercari marketplace](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/data). This model will regress the prices at which the products sold against these descriptions, creating a model that can generate a suggested price based off user inputs. With this "rough" price in mind, the seller can immediately gauge a good understanding of what their product will sell for, without the need to do any external market research.<br><br>

> __In one sentence: Can Transformers Accurately Decide an Items Price Given its Description__ ?? 

![Screenshot 2024-03-31 at 8 54 45 AM](https://github.com/edmundhhn/transformers-theory-and-practice/assets/97279107/15ad0037-367f-4a02-a985-d927dd4485c5)
![Screenshot 2024-03-31 at 8 58 44 AM](https://github.com/edmundhhn/transformers-theory-and-practice/assets/97279107/c046a244-b418-4f78-803c-39cffe542251)


## Dataset

Our [dataset](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/data) contains 1.4+ million listings of items being sold on Mercari. Each listing contains the following information

- __train_id or test_id__ - the id of the listing
- __name__ - the title of the listing. Note that we have cleaned the data to remove text that look like prices (e.g. $20) to avoid leakage. These removed prices are represented as [rm]
- __item_condition_id__ - the condition of the items provided by the seller
- __category_name__ - category of the listing
- __brand_name__ - The brand of the item being listed
- __price__ - the price that the item was sold for. This is the target variable that you will predict. The unit is USD. This column doesn't exist in test.tsv since that is what you will predict.
- __shipping__ - 1 if shipping fee is paid by seller and 0 by buyer
- __item_description__ - the full description of the item. Note that we have cleaned the data to remove text that look like prices (e.g. $20) to avoid leakage. These removed prices are represented as [rm]

## Project Steps

1. __Reformat Data into Single String__ To feed data into BERT, we extract relevant information from columns are combined into a single line of text. We set the target variable as the price of the item

![image](https://github.com/edmundhhn/transformers-theory-and-practice/assets/97279107/24bd0807-b838-492b-94e0-26a8179ac604)

2. __Simple preprocessing__ Remove mentions of price to avoid data leakage

3. __Create Dataset and DataLoader__ objects specific to our dataset, adding batch size of 32, shuffle during training. We train test validation split with __80/10/10__

4. Create a Class for a BERT Regressor which utilizes a regression head.
  __Recap:__ BERT is a a multi-layer bidirectional Transformer encoder. This architecture allows BERT to consider the full context of a word by looking at the words that come before and after it allowing it to understand the nuances of language.
   We add a linear layer at the end of the BERT model which uses the last layer weights of the BERT model to regress against a final price prediction

   - *Attention mask* to account for variable length strings (BERT takes same length)
   - *Dropout layer* (10%) to prevent overfitting 

   <img width="550" alt="BERT" align=”middle” src="https://github.com/edmundhhn/transformers-theory-and-practice/assets/97279107/4a6fa008-f6b6-4092-844b-8cb4dcd6540f">

7. __Tuning__ Select suitable hyperparameters and fine tune the model for 3 epochs on the data, recording the validation loss per epoch. 

8. __Testing__ Test the model on testing dataset and report metrics
   - Our final RMSE on the model is 11.22 with MAE of 9.51. Although this is not extremely accurate the seller can still gauge a rough idea of the price our item should be selling at

9. Finalize project with the front-end interface in Gradio.

---
## CODE and Gradio Demonstration 

---

## Critical Analysis

- __Impact__:
With the pricing model, sellers have a tool that can help them set a price easier, whilst buyers can also potentially use the tool to determine if the item they are purchasing is priced reasonably. 

- __Advantages__ :
  - __Streamlines the Pricing Process__: Our tool can save time and cost for both parties in deciding what is the optimum price for a product 
 
- __Disadvantages__ :
  - __Accuracy__: Should not be taken as the final pricing agent. There is still a significant gap between the prediction and actual price. There is heavy variation of a price of an item even if its the exact same - information imperfection. 
  - __Limitations of Data__:
      - Does not capture many aspects of a product that could impact its selling price, e.g. Seller reputation, age of item, rarity etc... Importantly, the image is not a part of the dataset which is a very critical identifier of item price
      - Historical data will not account/predict future trends
      - Data contains listings instead of actual items that sold. This includes items that did not sell too. 
  - __Limitations with Model__: A lot of our data is originally structured, which is somewhat under-utilized by a LLM such as BERT.

- __Next Steps__ :
  - __More, Varied Data__: For example we can include product images, utilizing a multimodal input for price prediction.
  - __Trends__: We need a model that has a deep understanding of whats "trendy" e.g. apple airpods, carhartt jackets
  - __Different Businesses__: We can attempt this method in other more stable marketplaces, for example that of Amazon and Walmart. These may be harder as the range of products they have are even broader. But the data will be broader.
  - __Enhanced Tools__: Given that we have a model that knows which listings sell for the highest price, can we now create a tool which can optimize descriptions for sellers to maximize their profits? Another idea is a system that can provide recommendations to sellers on what kind of items that they should procure (e.g. When at the bins at goodwill) in order to turn the greatest profits on their sales. 

## Resource Links

- __Mercari Platform__: [https://www.mercari.com/](https://www.mercari.com/)
- __Dataset__: [Mercari Price Suggestion Challenge](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/data)
- __BERT Original Paper__: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- __Another Sample of LLM Regression__: [Regression with Text Input Using BERT and Transformers](https://lajavaness.medium.com/regression-with-text-input-using-bert-and-transformers-71c155034b13)
  
