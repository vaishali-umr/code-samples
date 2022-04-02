####################################################
##                                                ##
##      Predicting Employee Termination w/        ##
##        Organizational HR Data                  ##
##                                                ##
####################################################

library(tidyverse)
library(GGally)
library(rsample)
library(caret)
library(InformationValue)
library(modelr)

setwd('~/Coding Samples/hr-data')



########## Introduction

# Can we predict who within our company is going to terminate?
# What level of accuracy can we achieve on this?

# If we can predict who might terminate, the company can consider investing its
# resources differently for those people.

hr <- read_csv('HRDataset_v14.csv')



########## Data Cleaning

# focus on variables of interest
hr_df <- hr %>% select(Salary, Department, EngagementSurvey, Absences, EmpSatisfaction, SpecialProjectsCount, DaysLateLast30, DateofHire, `LastPerformanceReview_Date`, DateofTermination, Termd)

# check for missing values
sum(is.na(hr_df))

# there are 207 missing values; let's see what column(s) they're in
colSums(is.na(hr_df))

# all 207 missing values are in DateofTermination which we'll handle later when
# feature engineering, so let's keep going for now

# range of each feature; check that min/max values make sense
# check for outliers
hr_df %>% dplyr::select(where(is.numeric)) %>% 
  summarise(across(everything(), range))

summary(hr_df)

# clean column names
hr_df <- rename(hr_df, c(`Employee Engagement` = EngagementSurvey, `Employee Satisfaction` = EmpSatisfaction, `Special Projects` = SpecialProjectsCount, `Days Late Last 30` = DaysLateLast30, Terminated = Termd))



########## Train/test Split

set.seed(42)

split <- initial_split(hr_df, prop = 0.7)
train <- training(split) # extract the actual data 
test <- testing(split) # extract the actual data

dim(split)



########## Exploratory Data Analysis (EDA)

# summary stats by termination status
train %>% group_by(Terminated) %>% dplyr::select(where(is.numeric)) %>% 
  summarise(across(everything(), mean))

# everything makes sense - avg salary, employee engagement, satisfaction, and
# special project count is lower for terminated folks than for those who stay.
# avg number of absences are lower for people that stay than for those terminated.

# continuous distribution - histogram
ggplot(data = train) +
  geom_histogram(mapping = aes(x = Salary))

ggplot(data = train) +
  geom_histogram(mapping = aes(x = `Employee Engagement`))

ggplot(data = hr_df) +
  geom_histogram(mapping = aes(x = `Absences`), binwidth=2)

ggplot(data = train) +
  geom_histogram(mapping = aes(x = `Employee Satisfaction`))

ggplot(data = train) +
  geom_histogram(mapping = aes(x = `Special Projects`), bins=10)

ggplot(data = train) +
  geom_histogram(mapping = aes(x = `Days Late Last 30`))

# categorical distribution - barplot
ggplot(data = train) +
  geom_bar(mapping = aes(x = Department))

# scatter plots for features of interest and termination status
ggplot(data = train) +
  geom_point(mapping = aes(x = Salary, y = Terminated), alpha = 0.3)

# it seems people with higher salaries are more likely to stay, which makes sense

ggplot(data = train) +
  geom_point(mapping = aes(x = `Employee Engagement`, y = Terminated), alpha = 0.3)

# strange that people with lower levels of engagement seem less likely to terminate

ggplot(data = train) +
  geom_point(mapping = aes(x = `Days Late Last 30`, y = Terminated), alpha = 0.2)

# needs more investigation, but more late days might be correlated with higher
# likelihood of termination

# pairplot using most interesting features
train %>% select(Salary, `Employee Engagement`, `Days Late Last 30`, Terminated) %>%
  ggpairs(train)



########## Feature Engineering

# interested in seeing if number of days worked is a good predictor of whether someone
# will leave the company - the longer people stay, do they become less likely to leave?

# build function to calculate number of days worked
calc_days_worked <- function(df) {

  # find most recent last performance review date
  recent_date <- max(as.Date(df$`LastPerformanceReview_Date`, format="%m/%d/%Y"))

  # create number of days worked column; delete unecessary columns
  df <- df %>% mutate(`Last Day Worked` = DateofTermination) %>%
    mutate(`Last Day Worked` = replace_na(`Last Day Worked`, format(recent_date, format="%m/%d/%Y"))) %>%
    mutate(`Days Worked`=as.integer(as.Date(`Last Day Worked`, format = "%m/%d/%Y") - as.Date(DateofHire, format = "%m/%d/%Y"))) %>%
    select(-c(DateofHire, LastPerformanceReview_Date, DateofTermination, `Last Day Worked`))

  return(df)
}

# add new feature to training data
train <- calc_days_worked(train)
  
# see if range of new feature makes sense
summary(train$`Days Worked`)

# min days worked is 26 and max is 4798, which seems plausible

# check distribution of new feature
ggplot(data = train) +
  geom_histogram(mapping = aes(x = `Days Worked`))

# plot new feature against termination status
ggplot(data = train) +
  geom_point(mapping = aes(x = `Days Worked`, y = Terminated), alpha = 0.2)

# those with fewer days under their belt seem more likely to terminate than more
# experienced employees, which intuitively makes sense

# add new feature to test data
test <- calc_days_worked(test)



########## Model

# Model 1
# logistic regression on training data without engineered feature
mod <- glm(Terminated~Salary+`Employee Engagement`+`Days Late Last 30`, family="binomial", data=train)

summary(mod)

# Model 2
# logistic regression on training data with engineered feature added
mod_feature <- glm(Terminated~Salary+`Employee Engagement`+`Days Late Last 30`+`Days Worked`, family="binomial", data=train)

summary(mod_feature)

# compare variable importance
varImp(mod)
varImp(mod_feature)

# in both models, employee engagement seems of least importance in predicting termination.
# salary and late days are of more importance, but number of days worked is the most
# useful feature as shown in Model 2.

# get predictions
predicted_without_feature <- predict(mod, test, type="response")
predicted_with_feature <- predict(mod_feature, test, type="response")

# confusion matrices
confusionMatrix(test$Terminated, predicted_without_feature)
confusionMatrix(test$Terminated, predicted_with_feature)

# at first glance, it looks like Model 2 with `Days Worked` added did better
# since the number of False Negatives and False Positives is lower - this means
# Model 2 resulted in fewer cases where an employee actually terminated but
# our model didn't catch it (False Negative) and where the employee wasn't terminated
# but our model predicted he or she would (False Positive).



########## Evaluate model accuracy

# precision
# precision tells us of everyone we predicted would terminate, how many actually did?
precision(test$Terminated, predicted_without_feature) #56%
precision(test$Terminated, predicted_with_feature) #66%

# Model 2 has a better precision rate of 66% (vs 56%).
# This means for every 100 terminations Model 2 predicts, 66 are actually terminations,
# and 34 aren't.


# sensitivity (recall) = true positive rate
# sensitivity tells us of all that terminated, how many did we predict correctly?
sensitivity(test$Terminated, predicted_without_feature) #18%
sensitivity(test$Terminated, predicted_with_feature) #75%

# Model 2 has better sensitivity of 75% (vs 18%).
# This means Model 2 correctly predicts 75% of terminations and misses 25% of cases.


# specificity = true negative rate
# specificity tells us of all employees that stayed, how many did we predict correctly?
specificity(test$Terminated, predicted_without_feature) #94%
specificity(test$Terminated, predicted_with_feature) #83%

# Model 1 has better specificity at 94% (vs 83%).
# This means Model 1 is right 94% of the time in predicting when an employee will stay,
# and incorrectly predicts that someone will terminate only 6% of the time.



########## Conclusion

# Overall, Model 2 (with engineered feature `Days Worked`) has better precision and 
# sensitivity. It seems to be a better model than Model 1 for predicting who will terminate.
# However, Model 1 showed a better level of specificity and might be useful in predicting
# employee retention. It would be useful to further break down this analysis by type of
# termination - voluntary vs for cause. Perhaps we can create separate models for predicting
# voluntary termination vs termination for cause. If we know who's at a higher risk of
# voluntarily leaving, we can allocate more resources towards retaining them. On the other
# hand, if we can predict who will be terminated for cause, we can invest fewer resources
# in that group of people.
