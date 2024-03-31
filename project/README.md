# Automated Pricing for Sellers on Mercari
__Authors: Thao Nguyen, Edmund Hui__

Play with our project on [Huggingface Spaces](https://huggingface.co/spaces/edmundhui/mercari-price-prediction) ! We take 25% comission for every successful sale using our application.

## Overview

- __Context__: Mercari is a Japanese e-comemrce company. Their main product, the Mercari marketplace app is a place for buyers and sellers to buy sell and trade items securely and hosts over 23 million users every year. 

- __Problem__: A major obstacle for the sellers on the Mercari marketplace is the difficulty of setting a competitive price for their product. Accurately predicting product prices is paramount for maximizing profits and staying ahead of the competition, whilst incorrectly setting prices could lead to revenue loss through over/under selling.

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
- __brand_name__
- __price__ - the price that the item was sold for. This is the target variable that you will predict. The unit is USD. This column doesn't exist in test.tsv since that is what you will predict.
- __shipping__ - 1 if shipping fee is paid by seller and 0 by buyer
- __item_description__ - the full description of the item. Note that we have cleaned the data to remove text that look like prices (e.g. $20) to avoid leakage. These removed prices are represented as [rm]

## Project Steps

1. Relevant information from columns are combined into a single line of text. Set the prices as target variable 

![image](https://github.com/edmundhhn/transformers-theory-and-practice/assets/97279107/24bd0807-b838-492b-94e0-26a8179ac604)

2. Simple preprocessing: Removing mentions of prices, and elements that may not be so relevant to the price.

3. Create Dataset and DataLoader objects specific to our dataset.

4. Create a Class for a BERT Regressor which utilizes a regression head.

   <img width="550" alt="BERT" align=”middle” src="https://github.com/edmundhhn/transformers-theory-and-practice/assets/97279107/4a6fa008-f6b6-4092-844b-8cb4dcd6540f">

6. Select suitable hyperparameters and fine tune the model for X epochs on the data, recording the validation loss per epoch. 

7. Test the model on testing dataset and report metrics

8. Finalize project with front-end interface

## Critical Analysis

- __Impact__:
With the pricing model, sellers have a tool that can help them set a price easier, whilst buyers can also potentially use the tool to determine if the item they are purchasing is priced reasonably. 

- __Advantages__ :
  - __Streamlines the Pricing Process__: Our tool can save time and cost for both parties in deciding what is the optimum price for a product 
 
- __Disadvantages__ :
  - __Accuracy__: Should not be taken as the final pricing agent. There is still a significant 
  - __Limitations of Data__:  Does not capture many aspects of a product that could impact its selling price, e.g. future trends, seller reputation, age of item, rarity etc... As we know in online shopping, the biggest determination of a product is often the image.
  - __Limitations with Model__: A lot of our data is originally structured, which is somewhat not utilized by a LLM such as BERT.

- __Next Steps__
  - __More, Varied Data__: For example we can include product images, utilizing a multimodal input for price prediction. 
  - __Different Businesses__: We can attempt this method in other marketplaces, for example that of Amazon and Walmart, these will be harder as the range of products they have are even broader.
  - __Description Generator and Additional Tools__: Given that we have a model that knows which listings sell for the highest price, can we now create a tool which can create descriptions for sellers to maximize their profits? Or another idea is a system that can provide recommendations to sellers on what kind of items that they should procure (e.g. When at the bins at goodwill) in order to turn the greatest profits on their sales. 

---
## CODE Demonstration
---

## Resource Links

- Mercari Platform: [https://www.mercari.com/](https://www.mercari.com/)
- Dataset: [Mercari Price Suggestion Challenge](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/data)
- BERT Original Paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- Another Sample of LLM Regression: [Regression with Text Input Using BERT and Transformers](https://lajavaness.medium.com/regression-with-text-input-using-bert-and-transformers-71c155034b13)
  
