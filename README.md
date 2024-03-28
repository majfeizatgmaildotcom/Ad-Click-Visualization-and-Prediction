# Ad-Click-Visualization-and-Prediction

<img src="Project_Dash2.gif" style>

**Project Status: Completed**
<br>
<a href="https://github.com/majfeizatgmaildotcom/Ad-Click-Visualization-and-Prediction/blob/eacfd418a8110df4fe3711b4d7f20083c4b42c3e/Visualization_project_forUCONN_PY_ver9_Final.py">Dashboard Python Viewer</a>
<br>
<a href="https://github.com/majfeizatgmaildotcom/Ad-Click-Visualization-and-Prediction/blob/e17aee6ff459dc43b05042ca7c1c34d209f2f2ae/AD%20Click%20Data%20Analaysis.ipynb">Jupyter Notebook Viewer For Utilization of GMM Analysis</a>
<br>
<a href="https://github.com/majfeizatgmaildotcom/Ad-Click-Visualization-and-Prediction/blob/9f147ee42b5dbde7d75935f7b28872423cc4a6da/Visualization_Presentation_Final_rev2.mp4">MP4 File for Dashboard Presentation</a>

## Project Objective
In the digital age, marketing has evolved, with companies increasingly leveraging online advertising. However, efficiently targeting the right audience is crucial to control advertising costs. Online marketing's challenge lies in accurately identifying and reaching potential customers. 

This project utilizes advertising marketing data to conduct comprehensive exploratory statistical and geo-mapping analyses. By analyzing key features like 'Age', 'Area Income', 'Daily Internet Usage', 'Ad Topic Line', 'City', 'Country', 'Gender', and 'Clicked on Ad', I aim to predict user engagement with ads. The focus is on the 'Clicked on Ad' variable, distinguishing between users who did (1) and did not (0) click on an ad.
<br>
## Methods Used
+ Data Collection
+ Data Cleaning
+ Exploratory Data Analysis
+ Data Visualization
+ Model Prediction

## Technologies
+ Python
+ NumPy 
+ Pandas 
+ Seaborn
+ Matplotlib
+ Dash plotly (for creating the dashboard)

## Dataset
 The dataset, sourced from Kaggle's advertising section, serves as the foundation for both exploratory statistical analysis and the training of a machine learning model. My initial goal is to undertake exploratory data analysis to understand the impact of variables like 'Daily Time Spent on Site' on the likelihood of a user clicking an advertisement. I aim to identify which customer demographics, based on their interests or gender, are more inclined to engage with ads. The ultimate objective is to develop a model capable of predicting the 'Clicked on Ad' outcome with acceptable accuracy.

## System Functionality
The dashboard is divided in the two-part as follows:\
Part 1: Exploratory Analysis\
Part 2:  Model Prediction

<!-- #### Part 1) In this part, the dashboard provides interactive figures so the user can do an exploratory statistical analysis of the data as follows:
1)	Entire plots in the dashboard will be interactively updated based on Age:\
Since the Age of the audience is fundamental, the user is given the capability to change the entire plot with a range of Age of the audience. The user can select the Age range that provides the best correlation amount other features.
2)	The entire plots in the dashboard are interactively updated based on a) the audience who clicked on the Ad, b) the audience who did not click on the Ad, c) both a and b groups.
3)	Geo map plot provided (figure 1) a country location of each data. This graph helps users identify which country has to most daily internet usage on the Site. The graph also calculated the average Age of the users in each country. For example, in this figure, for the user who clicked in Ad, the avg age in Turkey is about 41 years. It also shows Norway and Germany have the average Age of 52 are the oldest, and Japan, with avg Age of 24 is the youngest country.

4)	Treemap of Daily customer time Spent on Site in the figure shows the country, and the figure 2a, shows the city of each user. With this plot, a user can quickly see which county (and its cities) has the most daily time spent on Site. As the dashboard has interacted with age range and Clided Ad data, a user can see which city has the most daily time spent on the Site per selected age range. For example, it can be seen that Turkey has the highest daily spent on Sites, along with the city of Willemstad. By hovering the mouse, the user can also see the customer's Age who clicked on the Ad is around 30 in this city.

5) Scatter Matrix and Correlation plots in Figures 3, and 4 show the scattered and the correlation between each pair in the feature data. Figure 3 mainly indicates that the data can be clustered into two groups: audiences who clicked on Ads and those who did not click on Ad. 
By looking at both data sets in clicked Ad and non-clicked, in correlation plots in figure 4, we can see that generally speaking, the user who more spent on the Site are younger Age (negatively correlated), and the user who has more daily internet usage also has more spent on the Site.

6) Scatter plot of Daily Internet Usage vs. Daily Spent time on Site vs. Age, and its corresponding histogram and KDE demonstrated the distribution of Daily Internet Usage and Daily Time Spent on Site provided in Figure 5. All figures suggest that the data can be clustered in two groups with two different daily spent on-site time distributions to reduce the data to two groups.

7) The 2D density contour shows Daily Time Spent on Site and indicates that the data can be clustered in two groups in figure 6. This plot, along with GMM in figure 7, provided two group populations based on Daily Time Spent on Site vs. Age.

8) Gaussian Mixture Model method is used to see how the data can be clustered into two (or more) groups. Users can select the desired cluster, and the cluster shows with different colors. The Silhouettes method is recommended as an option for selecting the number of cluster groups. Figure 7 shows the scatter data of Daily Time Spent on Site vs. Age, with two symbols for Clicked on Ad 0, and 1.
Figure 8 and Figure 9 interactively show the corresponding cluster. Figure 8 shows the distribution in violin plots, and Figure 9 shows the KDE of the distribution. The p-values corresponding to each pair group are calculated based on the t-Student distribution to see if there is any significant difference in the two distributions (based on some significant level criteria). 
The two clusters group can give good criteria for reducing the data to two groups based on the Daily Time Spent on the Site. The idea would be to focus on the audience using less time on the Site and clicking on the Ad.

9) In this part of the dashboard, the gender impact in the data is analyzed. The null hypothesis is "The ave age of the male customer is the same as female customers". The alternative hypothesis is that avg Age is not the same.
Figure 10 on the left shows the distribution of males vs. females in the violin plot, which is very similar—in figure 10 on the right shows the corresponding KDE plot, which confirms the similar distribution. In the upper-right part of figure 10, the p-value of the two distributions has been calculated based on the t-Student method. As expected, the p-value is way larger than 5% significant interval level, so it is failed to reject the hypothesis, meaning there is no gender differences in using the Site.

#### Part 2) This part is dedicated to the model prediction:
A machine learning model is provided based on Logistic Regression with an accuracy of 90.6%. Figure 11 provides the performance of the model based on the confusion matrix. The total number of accurate predictions is 158+141 = 299, and the total number of incorrect predations is 27+4 = 31, which is a good performance measure for this application. -->

## Project Description
I have crafted a dashboard designed to enhance the targeting of marketing advertisements by aligning them with the audience's interests. The dashboard is split into two main sections: the first for exploratory statistical data analysis and the second for predictive modeling.

In the first section, users can access a variety of statistical analysis tools, including scatter matrix plots and Pearson correlation matrices to examine feature correlations, histograms with kernel density estimation for distribution analysis, and the Gaussian Mixture model for segmenting data based on Daily Internet Usage. T-tests and p-values are calculated for each cluster pair to explore distribution differences and test hypotheses, such as gender-based interest variances in site usage, with findings indicating no significant gender differences.

This dashboard offers insightful tools for analyzing ad interest disparities between genders, supported by hypothesis testing. It features interactive plots that adjust based on age range and whether an ad was clicked, providing a global Geo map and a tree map for visualizing user location data based on site engagement levels.

In the second section, a Logistic Regression model predicts audience likelihood to click on ads, with model performance showcased through accuracy metrics and a confusion matrix that adjusts based on age. The model demonstrates a satisfactory accuracy rate of 90.6% for this application, making it a valuable asset for targeting ad campaigns more effectively.
 
 ## Results
 The findings indicate that the site primarily attracts an adult demographic, with the average age of visitors ranging from 36 to 54 years (based on a +/-2 sigma level). There appears to be no significant gender difference in ad-clicking behavior, and a correlation exists between higher daily internet usage and increased time spent on the site. Notably, Turkey exhibits the highest average daily site engagement, particularly in the city of Willemstad, where the average visitor age is 30. Utilizing Logistic Regression for model predictions, I can determine the likelihood of a visitor clicking on an advertisement with a 90.6% accuracy rate. The dashboard’s interactive features, which adjust according to age range and ad-clicking data, serve as an effective tool for exploring various user interaction scenarios.

## More Detail
<a href="https://github.com/majfeizatgmaildotcom/Ad-Click-Visualization-and-Prediction/blob/e24fc35d8f38ecf143e2fc16e55e043232a5c723/Ad%20Click%20Visualization%20and%20Prediction%20Document.pdf">PDF Document Viewer</a>
<br>


