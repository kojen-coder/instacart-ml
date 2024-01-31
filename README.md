<h1>Predicting Instacart Customer Reorders</h1>
<p>This machine learning project's objective was to predict the likelihood of customers reordering products in their subsequent orders based on their historical purchase data. The dataset provided by Instacart includes over 3 million orders from more than 200,000 users, encompassing order history, product details, and purchasing patterns. The challenge was not only to accurately predict reorders but also to uncover insights into customer purchasing patterns.</p>

<h3>My Approach</h3>
<p>The process began with Exploratory Data Analysis (EDA), followed by using PySpark for data curation, which involved building cohorts and performing feature engineering. A baseline model was initially developed, which was progressively enhanced by integrating specific user and order features, markedly improving key performance indicators such as sensitivity, specificity, and accuracy. Further analysis on feature importance was conducted to identify key drivers of model performance. The refinement process led to the adoption of the XGBoost algorithm, which showed superior performance over logistic regression, particularly in sensitivity and AUC, underscoring its improved predictive precision and effectiveness.</p>

<h3>Repository Contents</h3>
<ul>
<li><b>notebooks/:</b> Contains Jupyter notebooks that document the exploratory data analysis, feature engineering process, and model development steps in detail.</li>
<li><b>src/:</b> Includes Python scripts for custom functions and utilities supporting data processing and model training.</li>
<li><b>data/:</b> Due to the competition's rules, the dataset is not provided within this repository. Access to the data can be obtained through the Kaggle competition page.</li>
<li><b>reports/:</b> Features visualizations, model performance metrics, prediction outputs, and business insights.</li></ul>
