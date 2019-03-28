getwd()

install.packages('R.utils')
library(R.utils)
library(data.table)
library(dplyr)
reviews <-  fread("reviews.csv.gz")
lists <- fread("listings.csv.gz")

#data cleaning

boxplot(lists$review_scores_rating, main = "Boxplot: Distribution of Review Score Ratings", ylab = "Rating Score")

levels <- c(-Inf, 50, 80, 90, 95, Inf)
labels <- c("1", "2", "3", "4", "5")
lists_cleaned <- lists %>%
                      select(id, review_scores_rating) %>%
                      mutate(ratings = cut(review_scores_rating, levels, labels = labels))


lists_cleaned$ratings <- as.factor(lists_cleaned$ratings)

summary(lists_cleaned$ratings)

lists_cleaned <- lists_cleaned[complete.cases(lists_cleaned),]

lists_cleaned <- as.data.frame(lists_cleaned)


reviews_cleaned <- merge(x = reviews, y = lists_cleaned, by.x = "listing_id", by.y = "id")

write.csv(reviews_cleaned, "reviews_cleaned.csv")

getwd()

