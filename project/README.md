# Automated Pricing for Sellers on Mercari

## Overview

- __Context__: Mercari is a Japanese e-comemrce company. Their main product, the Mercari marketplace app is a place for buyers and sellers to buy sell and trade items securely and hosts over 23 million users every year. 

- __Problem__: A major obstacle for the sellers on the Mercari marketplace is the difficulty of setting a competitive price for their product. Accurately predicting product prices is paramount for maximizing profits and staying ahead of the competition, whilst incorrectly setting prices could lead to revenue loss through over/under selling.

- __Approach__: To assist the seller in their price determination, we fine tune a BERT model on a large amount of user inputted descriptions of [products which sold on the Mercari marketplace](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/data). This model will regress the prices at which the products sold against these descriptions, creating a model that can generate a suggested price based off user inputs. With this "rough" price in mind, the seller can immediately gauge a good understanding of what their product will sell for, without the need to do any external market research.<br><br>

> __In one sentence: Can Transformers Accurately Decide an Items Price Given its Description__ ?? 

![image](https://github.com/edmundhhn/transformers-theory-and-practice/assets/97279107/bcbae617-2754-47b8-a409-1cc1abd777f7)
![image](https://github.com/edmundhhn/transformers-theory-and-practice/assets/97279107/86b070c0-ee60-48a3-ac8f-97c59e8ca5f2)


## Critical Analysis

Through our project 

- __Impact__:
  - __Pricing Assisstant__: With the pricing model, sellers have a tool that can help them set a price easier, whilst buyers can also potentially use the tool to determine if the item they are purchasing is priced reasonably. 
- __Main Revelations__ :
  - __Predictive Power__: We found that product description alone can have significant predictive power on an items price
  - __Use with Caution__: The RMSE suggests that this should not be a be all end all for pricing. Platform users should still use their own judgement whilst setting a final price 
- __Next Steps__
  - __More Data__: For example we can include product images, utilizing a multimodal input for price prediction. 
  - __Different Businesses__: We can attempt this method in other marketplaces, for example that of Amazon and Walmart, these will be harder as the range of products they have are even broader.

---
## CODE Demonstration
---

## Resource Links

- Mercari Platform: [https://www.mercari.com/](https://www.mercari.com/)
- Dataset: [Mercari Price Suggestion Challenge](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/data)
- BERT Original Paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- Another Sample of LLM Regression: [Regression with Text Input Using BERT and Transformers](https://lajavaness.medium.com/regression-with-text-input-using-bert-and-transformers-71c155034b13)
  
