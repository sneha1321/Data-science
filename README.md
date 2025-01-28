# Data-science

# **eCommerce Transactions - Data Science Assignment**  

## **About This Project**  
This project is part of a Data Science Internship assignment. The goal is to analyze an eCommerce transactions dataset, build a lookalike model, and perform customer segmentation using machine learning techniques.  

## **Dataset Details**  
The dataset includes three files:  
- **Customers.csv** – Information about customers (ID, Name, Region, Signup Date).  
- **Products.csv** – Product details (ID, Name, Category, Price).  
- **Transactions.csv** – Transaction records (ID, Customer ID, Product ID, Date, Quantity, Total Value, Price).  

## **Project Tasks**  
### **1. Exploratory Data Analysis (EDA)**  
- Analyzed the dataset to find trends, patterns, and insights.  
- Created visualizations to understand customer behavior and product sales.  
- Deliverables: **Jupyter Notebook + PDF Report with insights**.  

### **2. Lookalike Model**  
- Built a machine learning model to find similar customers based on their profile and purchases.  
- Recommended **3 similar customers** for each of the first 20 customers.  
- Deliverables: **Jupyter Notebook + Lookalike.csv file**.  

### **3. Customer Segmentation (Clustering)**  
- Grouped customers into different segments using clustering techniques.  
- Evaluated clusters using the **DB Index** and created visualizations.  
- Deliverables: **Jupyter Notebook + PDF Report**.  

## **Technologies Used**  
- **Python** (pandas, numpy, matplotlib, seaborn, scikit-learn)  
- **Jupyter Notebook**  
- **Machine Learning (Clustering & Similarity Models)**  

## **How to Run the Code**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/repository-name.git
   ```  
2. Install required libraries:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Open **Jupyter Notebook** and run the following files in order:  
   - `EDA.ipynb`  
   - `Lookalike_Model.ipynb`  
   - `Customer_Segmentation.ipynb` 

