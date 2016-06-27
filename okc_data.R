###################################################################
## Code for the useR 2016 tutorial "Never Tell Me the Odds! Machine 
## Learning with Class Imbalances" by Max Kuhn
## 
## Slides and this code can be found at
##    https://github.com/topepo/useR2016
## 
## packages used here are: lubridate
## 
## Data are at: https://github.com/rudeboybert/JSE_OkCupid

library(lubridate)

###################################################################
## Some levels of predictors have spaces or symbols; fix these
## and replace "" with "missing"

fix_levels <- function(x) {
  x <- gsub("&rsquo;", "", x) 
  x <- gsub("[[:space:]]", "_", x)
  x <- gsub("[[:punct:]]", "_", x) 
  x <- gsub("__", "_", x, perl = TRUE)
  x <- gsub("__", "_", x, perl = TRUE)
  x <- gsub("__", "_", x, perl = TRUE)
  x[x == ""] <- "missing"
  resort_lvl(x)
}

###################################################################
## resort categorical predictors so that "missing" is the first level
resort_lvl <- function(x) {
  x <- as.character(x)
  lv <- c("missing", sort(unique(x[x != "missing"])))
  factor(x, levels = lv)
}

###################################################################
## Unpackage the data and read in. A compressed version of the csv
## file is at https://github.com/rudeboybert/JSE_OkCupid

raw <- read.csv("profiles.csv", 
                stringsAsFactors = FALSE)
raw <- raw[, !grepl("^essay", names(raw))]

###################################################################
## Compute the number of days since last online

tmp_last <- ymd(substring(raw$last_online, 1, 10))
tmp_last <- difftime(max(tmp_last), tmp_last, units = "days")
raw$last_online <- as.numeric(tmp_last)

###################################################################
## encode the "easy" categorical predictors

raw$body_type <- fix_levels(raw$body_type)
raw$drinks <- fix_levels(raw$drinks)
raw$drugs <- fix_levels(raw$drugs)
raw$education <- fix_levels(raw$education)
raw$diet <- fix_levels(raw$diet)
raw$job <- fix_levels(raw$job)
raw$offspring <- fix_levels(raw$offspring)
raw$pets <- fix_levels(raw$pets)
raw$orientation <- fix_levels(raw$orientation)
raw$sex <- fix_levels(raw$sex)
raw$smokes <- fix_levels(raw$smokes)
raw$drugs <- fix_levels(raw$drugs)
raw$status <- fix_levels(raw$status)

###################################################################
## Income is basically encoded categorical so we will make it a factor

test <- ifelse(raw$income == -1, NA, raw$income)
test <- factor(paste0("inc", test), levels = c("missing", paste0("inc", sort(unique(test)))))
test[is.na(test)] <- "missing"
raw$income <- test

###################################################################
## Split their location into city and state. There are some R functions
## (ahem, randomForest) that can only handle predictors with <=52
## levels so we take the long tail of the distribution and truncate
## some cities to "other"

tmp_where <- strsplit(raw$location, split = ", ")
where_state <- unlist(lapply(tmp_where, function(x) if(length(x) == 2) x[2] else "missing"))
where_town <- unlist(lapply(tmp_where, function(x) if(length(x) == 2) x[1] else "missing"))

town_tab <- sort(-table(where_town))
where_town[!(where_town %in% names(town_tab)[1:50])] <- "other"

raw$where_state <- factor(gsub(" ", "_", where_state))
raw$where_town <- factor(gsub(" ", "_", where_town))
raw$location <- NULL

###################################################################
## Some predictors have values and modifiers that describe how
## serious they are about their choice. We will create predictors
## for both characteristics of their answer

## for religon, split religon and modifier
tmp_relig_split <- strsplit(raw$religion, split = " ")
tmp_relig <- unlist(lapply(tmp_relig_split, function(x) x[1]))
tmp_relig[tmp_relig == ""] <- "missing"
tmp_relig[is.na(tmp_relig)] <- "missing"
raw$religion <- resort_lvl(tmp_relig)
raw$religion_modifer <- unlist(lapply(tmp_relig_split, 
                                      function(x) 
                                        if(length(x) > 1) 
                                          paste(x[-1], collapse = "_") else 
                                            "missing"))
raw$religion_modifer <- resort_lvl(raw$religion_modifer)

###################################################################
## Same for sign

raw$sign <- gsub("&rsquo;", "", raw$sign)
tmp_sign_split <- strsplit(raw$sign, split = " ")
tmp_sign <- unlist(lapply(tmp_sign_split, function(x) x[1]))
sign_lvl <- sort(unique(tmp_sign))
tmp_sign[tmp_sign == ""] <- "missing"
tmp_sign[is.na(tmp_sign)] <- "missing"
raw$sign <- resort_lvl(tmp_sign)
raw$sign_modifer <- unlist(lapply(tmp_sign_split, 
                                  function(x) 
                                    if(length(x) > 1) 
                                      paste(x[-1], collapse = "_") else 
                                        "missing"))
raw$sign_modifer <- resort_lvl(raw$sign_modifer)

###################################################################
## They are allowed to list multiple languages so we will pre-split
## these into dummy variables since they might have multiple choices.
## Also, "c++" and "lisp" !

tmp_speaks <- gsub("(", "", raw$speaks, fixed = TRUE)
tmp_speaks <- gsub(")", "", tmp_speaks, fixed = TRUE)
tmp_speaks <- gsub("c++", "cpp", tmp_speaks, fixed = TRUE)
tmp_speaks_split <- strsplit(tmp_speaks, split = ",")
tmp_speaks_split <- lapply(tmp_speaks_split, 
                           function(x) gsub("^ ", "", x))
tmp_speaks_split <- lapply(tmp_speaks_split, 
                           function(x) gsub(" ", "_", x))
speaks_values <- sort(unique(unlist(tmp_speaks_split)))
# tmp_speaks <- unlist(lapply(tmp_speaks_split, paste, collapse = ",", sep = ""))
for(i in speaks_values) 
  raw[, i] <- ifelse(unlist(lapply(tmp_speaks_split, function(x, sp) any(x == sp), sp = i)), 1, 0)
raw$speaks <- NULL

###################################################################
## Similaly, ethnicity is pre-split into dummy variables

tmp_eth <- gsub(", ", ",", raw$ethnicity)
tmp_eth <- gsub("/ ", "", tmp_eth)
tmp_eth <- gsub(" ", "_", tmp_eth)
tmp_eth_split <- strsplit(tmp_eth, split = ",")
eth_lvl <- sort(unique(unlist(tmp_eth_split)))
for(i in eth_lvl) 
  raw[, i] <- ifelse(unlist(lapply(tmp_speaks_split, function(x, eth) any(x == eth), eth = i)), 1, 0)
raw$ethnicity <- NULL

###################################################################
## There are very few missing values for continuous fields so 
## remove them and convert the job field to the outcome. 

okc <- raw[complete.cases(raw),]

okc$Class <- factor(ifelse(grepl("(computer)|(science)", okc$job), "stem", "other"),
                    levels = c("stem", "other"))


okc <- okc[okc$job != "missing",]
okc$job <- NULL

table(okc$Class)/nrow(okc)

save(okc, file = "okc.RData")
