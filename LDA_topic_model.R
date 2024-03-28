####################################################################
# This script is for Topic modelling                               #
#                                                                  #                                                         
#                                                                  #                                                  
# Author: Mathias Bellat                                           #
# Affiliation : Tübingen University                                #
# Creation date : 28/03/2024                                       #
# E-mail: mathias.bellat@uni-tubingen.de                           #
####################################################################

# 1 Preparation ####################################################################
# 1.1 Prepare workingspace  ---------------------------------------------------------------

# Folder check
getwd()
setwd("path/")


# Load packages
install.packages("pacman")
library(pacman) #Easier way of loading packages
pacman::p_load(tm, textmineR, udpipe, ldatuning, lda, LDAvis, readr, servr)

# 1.2 Import data ---------------------------------------------------------------
filtered_data <- read_csv("./Data/Data.csv")


# 1.3 Clean and prepare the DTM  ---------------------------------------------------------------
stop_words <- stopwords("SMART") #common stop words
stop_words <- c("al", "oa","fig", "sr","archaeological", "archaeology","set","data", "doi", "https", "wiley", "onlinelibrarywileycom",stop_words) #specifics words spot after a first run

# 1.4 Clean and prepare the corpus  ---------------------------------------------------------------
text <- as.character(filtered_data$Abstract)

# pre-processing (possibly alreay cleened before):
text <- gsub("'", "", text)  # remove apostrophes
text <- gsub("[[:punct:]]", " ", text)  # replace punctuation with space
text <- gsub("[[:cntrl:]]", " ", text)  # replace control characters with space
text <- gsub("^[[:space:]]+", "", text) # remove whitespace at beginning of documents
text <- gsub("[[:space:]]+$", "", text) # remove whitespace at end of documents
text <- gsub("[0-9.]", "", text) # remove numbers in the document
text <-  removeWords(text, stop_words)#Remove stopwords
text <- gsub("  ", " ", text)  # remove double spaces has to be process at least 3 times
text <- gsub("  ", " ", text)  # remove double spaces
text <- gsub("  ", " ", text)  # remove double spaces
text <- tolower(text)  # force to lowercase

export <- as.data.frame(text)
export <- cbind(filtered_data$Id, export)
write.csv(export, "./Data_pre_process.csv")

# tokenize on space and output as a list:
doc.list <- strsplit(text, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)

# remove terms that are stop words or occur fewer than 5 times:
del <- names(term.table) %in% stop_words | term.table < 5 

term.table <- term.table[!del]
vocab <- names(term.table)

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)

# 2 Latent Dirichlet Allocation ####################################################################

# 2.1 LDA Best parameters ---------------------------------------------------------------

system.time(optimal.topics <- FindTopicsNumber(dtm, 
                topics = seq(10,25, by=1),
                metrics = c("Griffiths2004", "Arun2010", "Deveaud2014"),
                method = "Gibbs")) # Find the best number of Topics for LDA

FindTopicsNumber_plot(optimal.topics) # Plot the Differents metrics and number of topics for LDA
save(optimal.topics,file = "./Export/TM/LDA_parameters_abstract.RData")

# 2.2 LDA Hyperparameters ---------------------------------------------------------------

# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (141)
W <- length(vocab)  # number of terms in the vocab (919)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document 
N <- sum(doc.length)  # total number of tokens in the data (384,190)
term.frequency <- as.integer(term.table)  # frequencies of terms in the corpus

# MCMC and model tuning parameters:
K <- 16
G <- 5000
alpha <- 0.01
eta <- 0.02

# 2.3 LDA Model running ---------------------------------------------------------------

# Fit the model:
set.seed(1070)
t1 <- Sys.time()
model_lda <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
t2 - t1  # compute time of runing


theta <- t(apply(model_lda$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(model_lda$topics) + eta, 2, function(x) x/sum(x)))


LDA_article <- list(phi = phi,
                    theta = theta,
                    doc.length = doc.length,
                    vocab = vocab,
                    term.frequency = term.frequency)


# 2.4 LDA results visualisation ---------------------------------------------------------------

# create the JSON object to feed the visualization:
json <- createJSON(phi = LDA_article$phi, 
                   theta = LDA_article$theta, 
                   doc.length =LDA_article$doc.length, 
                   vocab = LDA_article$vocab, 
                   term.frequency = LDA_article$term.frequency)

serVis(json, out.dir = './16__abs', open.browser = TRUE) # outputs lda visualization

save(json, LDA_article, model_lda, documents, doc.list, file = "./Data/LDA_abstract_16.RData") 