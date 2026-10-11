#!/usr/bin/env Rscript

tr_df <- function(groups) {
  groups - 1
}

e_df <- function(groups, sample_size) {
  (groups * sample_size) - groups
}

total_df <- function(groups, sample_size) {
  e_df(groups, sample_size) + tr_df(groups)
}

mse <- function(f_obs, ol_t, groups, sample_size) {
  ol_t / (f_obs * tr_df(groups) + e_df(groups, sample_size))
}

mst <- function(f_obs, mse) {
  f_obs * mse
}

sstr <- function(groups, sample_size, variance, f_obs) {
  ol_t <- sst(groups, sample_size, variance)
  mse <- mse(f_obs, ol_t, groups, sample_size)
  mst <- mst(f_obs, mse)

  mst * tr_df(groups)
}

sse <- function(groups, sample_size, variance, f_obs) {
  ol_t <- (variance^2) * (sample_size * groups - 1)
  ol_t - sstr(groups, sample_size, variance, f_obs)
}

sst <- function(groups, sample_size, variance) {
  (variance^2) * (sample_size * groups - 1)
}

sstr(4, 11, 2, 0.366)
sse(4, 11, 2, 0.366)
groups <- 4
sample_size <- 11
ol_t <- sst(4, 11, 2)

f_obs <- 0.366
ms <- mse(f_obs, ol_t, groups, sample_size)
mst(0.366, ms)
ms
