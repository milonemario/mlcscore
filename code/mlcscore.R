library(stats)
library(Hmisc)
library(arrow)
library(data.table)
library(stargazer)
library(multiwayvcov)
library(car)

# Folder
folder = 'results_06-21-2021'
folder_tables = paste0(folder, '/tables/')

# Open data
data <- read_parquet(paste0('data/data_',folder,'.parquet'))
data <- data.table(data)

# # Add previous C scores for debug
# data_old <- fread('data/data_results_03-30-2021.csv')
# cscores_names_old = c('c_L_basic', 'c_Lfmy_basic', 'c_NN_basic', 'c_NNy_basic',
#                       'c_L_kw', 'c_Lfmy_kw', 'c_NN_kw', 'c_NNy_kw')
# data_old = data_old[, c('gvkey', 'fyear', cscores_names_old), with=F]
# data <- merge(data, data_old, by=c('gvkey', 'fyear'))

# Standardize the C scores
cscores_names = c('L1y', 'MLC1', 'MLC1y', 'L2y', 'MLC2', 'MLC2y')
# Debug (try with averages and not final models)
#cscores_names = c('L1y', 'MLC1_mean', 'MLC1y_mean', 'L2y', 'MLC2_mean', 'MLC2y_mean')
for (cscore_name in cscores_names) {
  data[, (cscore_name) := get(cscore_name) / sd(get(cscore_name), na.rm=T)]
}


# # Add Jeremy crash data (DEBUG)
# dataj <- fread('data/data_jeremy_crash.csv')
# dataj[, permno := k_permno]
# dataj[, date := as.Date(k_datadate, '%d%B%Y')]
# data[, date := as.Date(datadate)]
# data <- merge(data, dataj[,list(permno, date, f_crash)], all.x=T, all.y=F, by=c('permno', 'date'))

##############################################
# Crash Risk - based on Kim and Zhang (2016) #
##############################################

# Create lead variables for crash, ncskew and roa
data <- data[order(gvkey, datadate)]
#data[, c('crash1', 'ncskew1', 'roa1', 'f_crash1') := shift(.SD, type='lead'), by=gvkey, .SDcols=c('crash', 'ncskew', 'roa', 'f_crash')]
data[, c('crash1', 'ncskew1', 'roa1') := shift(.SD, type='lead'), by=gvkey, .SDcols=c('crash', 'ncskew', 'roa')]

# Table 5
models = list()
models_se = list()
#modelsj = list()
# Select subsample
#df <- data[fyear>=1990 & fyear<=2007,]
df <- data
#cscores_names <- cscores_names_old
for (cscore_name in cscores_names) {
  df[, cscore := get(cscore_name)]
  cat(paste('# ',cscore_name,'\n'))
  cat('table 5 Panel A\n')
  m <- glm(crash1 ~ cscore + turnover + ncskew + yvolatd + yretm + logeqdef + m_b + flev + roa1 + factor(fyear), data=df, family='binomial')
  #m <- glm(f_crash1 ~ cscore + duvol + ncskew + yvolatd + yretm + logeqdef + m_b + flev + roa1 + factor(fyear), data=df, family='binomial')
  #m <- lm(crash1 ~ cscore + turnover + ncskew + yretm + m_b + roa1 + yvolatd + logeq + flev + factor(fyear), data=df, family='binomial')
  vcov <- cluster.vcov(m, cbind(df$gvkey, df$fyear))
  robust_se <- sqrt(diag(vcov))
  
  # DEBUG #
  # keep = c('cscore', 'turnover', 'ncskew', 'yvolatd', 'yretm', 'logeqdef', 'm_b', 'flev', 'roa1')
  # stargazer(m, se=list(robust_se),
  #           keep=keep, column.labels=cscores_names, type='text')
  
  models[['table5A']][[cscore_name]] <- m
  models_se[['table5A']][[cscore_name]] <- robust_se
  
  cat('table 5 Panel C\n')
  m <- lm(ncskew1 ~ cscore + turnover + ncskew + yvolatd + yretm + logeqdef + m_b + flev + roa1 + factor(fyear), data=df)
  vcov <- cluster.vcov(m, cbind(df$gvkey, df$fyear))
  robust_se <- sqrt(diag(vcov))
  
  models[['table5C']][[cscore_name]] <- m
  models_se[['table5C']][[cscore_name]] <- robust_se
  
  # DEBUG #
  # stargazer(m, se=list(robust_se),
  #           keep=keep, column.labels=cscores_names, type='text')
  
  # print(paste('Logit f_crash', cscore_name))
  # mj <- glm(f_crash1 ~ cscore + turnover + ncskew + yvolatd + yretm + logeqdef + m_b + flev + roa1 + factor(fyear), data=data, family='binomial')
  # modelsj[[cscore_name]] <- mj
}

# # DEBUG
# keep = c('cscore')
# labels = c('C score')
# stargazer(models[['table5A']],
#           keep=keep, covariate.labels=labels,
#           keep.stat=c('rsq', 'n'),
#           type='text')

keep = c('cscore', 'turnover', 'ncskew', 'yvolatd', 'yretm', 'logeqdef', 'm_b', 'flev', 'roa1')
labels = c('C score', 'Turnover', 'NCSKEW', 'Volatility', 'Return', 'Size', 'M/B ratio', 'Leverage', '$ROA_{t+1}$')

#keep = c('cscore', 'duvol', 'ncskew', 'yvolatd', 'yretm', 'logeqdef', 'm_b', 'flev', 'roa1')
#labels = c('C score', 'DUVOL', 'NCSKEW', 'Volatility', 'Return', 'Size', 'M/B ratio', 'Leverage', '$ROA_{t+1}$')

# Table 5A
stargazer(models[['table5A']], se=models_se[['table5A']],
          dep.var.labels='$Crash_{t+1}$', keep=keep, covariate.labels=labels,
          keep.stat=c('rsq', 'n'), column.labels=cscores_names,
          type='latex', float=F, out=paste0(folder_tables,'KimZhang_crash.tex'))
          #type='text')

# Table 5C
stargazer(models[['table5C']], se=models_se[['table5C']],
          dep.var.labels='$NCSKEW_{t+1}$', keep=keep, covariate.labels=labels,
          keep.stat=c('adj.rsq', 'n'), column.labels=cscores_names,
          type='latex', float=F, out=paste0(folder_tables,'KimZhang_ncskew.tex'))
          #type='text')


#########################
# Investment Efficiency #
#########################

dataie <- fread('data/data_LaraEtAl.csv')

dataie <- merge(dataie, data[,c('gvkey', 'fyear', cscores_names), with=F], all.x=T, all.y=F, by.x=c('gvkey', 'yeara'), by.y=c('gvkey', 'fyear'))

#dataie <- merge(dataie, data[,list(gvkey, fyear, L1, L1y, MLC1, MLC1y, L2, L2y, MLC2, MLC2y)], all.x=T, all.y=F, by.x=c('gvkey', 'yeara'), by.y=c('gvkey', 'fyear'))

# Define formulas for tables 2 and 3
formula2 <- as.formula('futinvestment ~ CO + CO_underinvest + FRQ + FRQ_underinvest + 
                        insholdings + insholdings_underinvest + analysts +
                        analysts_underinvest + invgindex + invgindex_underinvest +
                        dgindex + underinvest + IA + ncr + young +
                        size + mtbe + levmve + acdep + stdcfo + stdsale +
                        stdinves + zscore + tangibility + indkstructure + cfosale +
                        dividend + opercycle + invcycle + loss + slack +
                        factor(yeara)')

formula3IA <- as.formula('futinvestment ~ CO + CO_IA + CO_underinvest + CO_underinvest_IA +
                          FRQ + FRQ_IA + FRQ_underinvest + FRQ_underinvest_IA + IA_underinvest +
                          insholdings + insholdings_underinvest + analysts + analysts_underinvest +
                          invgindex + invgindex_underinvest + dgindex + underinvest + IA + ncr + young +
                          size + mtbe + levmve + acdep + stdcfo + stdsale + stdinves + zscore + tangibility +
                          indkstructure + cfosale + dividend + opercycle + invcycle + loss + slack +
                          factor(yeara)')

formula3NCR <- as.formula('futinvestment ~ CO + CO_ncr + CO_underinvest + CO_underinvest_ncr +
                           FRQ + FRQ_ncr + FRQ_underinvest + FRQ_underinvest_ncr + ncr_underinvest +
                           insholdings + insholdings_underinvest + analysts + analysts_underinvest +
                           invgindex + invgindex_underinvest + dgindex + underinvest + IA + ncr + young +
                           size + mtbe + levmve + acdep + stdcfo + stdsale + stdinves + zscore + tangibility +
                           indkstructure + cfosale + dividend + opercycle + invcycle + loss + slack +
                           factor(yeara)')

formula3YOUNG <- as.formula('futinvestment ~ CO + CO_young + CO_underinvest + CO_underinvest_young +
                             FRQ + FRQ_young + FRQ_underinvest + FRQ_underinvest_young + young_underinvest +
                             insholdings + insholdings_underinvest + analysts + analysts_underinvest +
                             invgindex + invgindex_underinvest + dgindex + underinvest + IA + ncr + young +
                             size + mtbe + levmve + acdep + stdcfo + stdsale + stdinves + zscore + tangibility +
                             indkstructure + cfosale + dividend + opercycle + invcycle + loss + slack +
                             factor(yeara)')


#cscores_names = c('L1', 'L1y', 'MLC1', 'MLC1y', 'L2', 'L2y', 'MLC2', 'MLC2y')
#cscores_names = c('CO') # DEBUG: to check with paper's results
models = list()
models_se = list()
models_tstat = list()
models_pvalue = list()
for (cscore_name in cscores_names) {
  cat(paste('#',cscore_name,'\n'))
  
  # Compute interactions
  dataie[, CO := get(cscore_name)]
  dataie[, CO_IA := CO*IA]
  dataie[, CO_ncr := CO*ncr]
  dataie[, CO_young := CO*young]
  dataie[, CO_underinvest := CO*underinvest]
  dataie[, CO_underinvest_IA := CO*underinvest*IA]
  dataie[, CO_underinvest_ncr := CO*underinvest*ncr]
  dataie[, CO_underinvest_young := CO*underinvest*young]
  
  ###########
  # Table 2 #
  ###########
  cat('Table 2\n')
  m <- lm(formula2, data=dataie)
  vcov <- cluster.vcov(m, cbind(dataie$gvkey, dataie$yeara))
  robust_se <- sqrt(diag(vcov))
  robust_tstat <- m$coefficients / robust_se
  robust_pvalue <- 2*pt(-abs(robust_tstat), df=length(dataie)-1)
  # Use a random unused covariate to display the sum of coefs (i.e. dividend)
  sumcoefs <- m$coefficients[['CO']] + m$coefficients[['CO_underinvest']]
  lh <- linearHypothesis(m, c("CO + CO_underinvest=0"), vcov.=vcov)
  #tstat <- sign(sumcoefs) * sqrt(lh$F[2])
  tstat <- sqrt(lh$F[2])
  se <- abs(sumcoefs / tstat)
  pvalue <- 2*pt(-abs(tstat), df=length(dataie)-1)
  m$coefficients[['dividend']] = sumcoefs
  robust_se[['dividend']] <- se
  robust_pvalue[['dividend']] <- pvalue
  robust_tstat[['dividend']] <- tstat
  # Add to the list of results
  models[['table2']][[cscore_name]] <- m
  models_se[['table2']][[cscore_name]] <- robust_se
  models_pvalue[['table2']][[cscore_name]] <- robust_pvalue
  models_tstat[['table2']][[cscore_name]] <- robust_tstat
  
  ################
  # Table 3 - IA #
  ################
  cat('Table 3 - IA\n')
  m <- lm(formula3IA, data=dataie)
  vcov <- cluster.vcov(m, cbind(dataie$gvkey, dataie$yeara))
  robust_se <- sqrt(diag(vcov))
  robust_tstat <- m$coefficients / robust_se
  robust_pvalue <- 2*pt(-abs(robust_tstat), df=length(dataie)-1)
  # Use a random unused covariate to display the sum of coefs (i.e. dividend)
  sumcoefs <- m$coefficients[['CO']] + m$coefficients[['CO_IA']]
  lh <- linearHypothesis(m, c("CO + CO_IA=0"), vcov.=vcov)
  tstat <- sqrt(lh$F[2])
  se <- abs(sumcoefs / tstat)
  pvalue <- 2*pt(-abs(tstat), df=length(dataie)-1)
  m$coefficients[['dividend']] = sumcoefs
  robust_se[['dividend']] <- se
  robust_pvalue[['dividend']] <- pvalue
  robust_tstat[['dividend']] <- tstat
  # Sum of all 4 coefficients
  sumcoefs <- m$coefficients[['CO']] + m$coefficients[['CO_IA']] + m$coefficients[['CO_underinvest']] + m$coefficients[['CO_underinvest_IA']]
  lh <- linearHypothesis(m, c("CO + CO_IA + CO_underinvest + CO_underinvest_IA = 0"), vcov.=vcov)
  tstat <- sqrt(lh$F[2])
  se <- abs(sumcoefs / tstat)
  pvalue <- 2*pt(-abs(tstat), df=length(dataie)-1)
  m$coefficients[['opercycle']] = sumcoefs
  robust_se[['opercycle']] <- se
  robust_pvalue[['opercycle']] <- pvalue
  robust_tstat[['opercycle']] <- tstat
  # Add to the list of results
  models[['table3IA']][[cscore_name]] <- m
  models_se[['table3IA']][[cscore_name]] <- robust_se
  models_pvalue[['table3IA']][[cscore_name]] <- robust_pvalue
  models_tstat[['table3IA']][[cscore_name]] <- robust_tstat
  
  ################
  # Table 3 - NCR #
  ################
  cat('Table 3 - NCR\n')
  m <- lm(formula3NCR, data=dataie)
  vcov <- cluster.vcov(m, cbind(dataie$gvkey, dataie$yeara))
  robust_se <- sqrt(diag(vcov))
  robust_tstat <- m$coefficients / robust_se
  robust_pvalue <- 2*pt(-abs(robust_tstat), df=length(dataie)-1)
  # Use a random unused covariate to display the sum of coefs (i.e. dividend)
  sumcoefs <- m$coefficients[['CO']] + m$coefficients[['CO_ncr']]
  lh <- linearHypothesis(m, c("CO + CO_ncr=0"), vcov.=vcov)
  tstat <- sqrt(lh$F[2])
  se <- abs(sumcoefs / tstat)
  pvalue <- 2*pt(-abs(tstat), df=length(dataie)-1)
  m$coefficients[['dividend']] = sumcoefs
  robust_se[['dividend']] <- se
  robust_pvalue[['dividend']] <- pvalue
  robust_tstat[['dividend']] <- tstat
  # Sum of all 4 coefficients
  sumcoefs <- m$coefficients[['CO']] + m$coefficients[['CO_ncr']] + m$coefficients[['CO_underinvest']] + m$coefficients[['CO_underinvest_ncr']]
  lh <- linearHypothesis(m, c("CO + CO_ncr + CO_underinvest + CO_underinvest_ncr = 0"), vcov.=vcov)
  tstat <- sqrt(lh$F[2])
  se <- abs(sumcoefs / tstat)
  pvalue <- 2*pt(-abs(tstat), df=length(dataie)-1)
  m$coefficients[['opercycle']] = sumcoefs
  robust_se[['opercycle']] <- se
  robust_pvalue[['opercycle']] <- pvalue
  robust_tstat[['opercycle']] <- tstat
  # Add to the list of results
  models[['table3NCR']][[cscore_name]] <- m
  models_se[['table3NCR']][[cscore_name]] <- robust_se
  models_pvalue[['table3NCR']][[cscore_name]] <- robust_pvalue
  models_tstat[['table3NCR']][[cscore_name]] <- robust_tstat
  
  ###################
  # Table 3 - YOUNG #
  ###################
  cat('Table 3 - YOUNG\n')
  m <- lm(formula3YOUNG, data=dataie)
  vcov <- cluster.vcov(m, cbind(dataie$gvkey, dataie$yeara))
  robust_se <- sqrt(diag(vcov))
  robust_tstat <- m$coefficients / robust_se
  robust_pvalue <- 2*pt(-abs(robust_tstat), df=length(dataie)-1)
  # Use a random unused covariate to display the sum of coefs (i.e. dividend)
  sumcoefs <- m$coefficients[['CO']] + m$coefficients[['CO_young']]
  lh <- linearHypothesis(m, c("CO + CO_young=0"), vcov.=vcov)
  tstat <- sqrt(lh$F[2])
  se <- abs(sumcoefs / tstat)
  pvalue <- 2*pt(-abs(tstat), df=length(dataie)-1)
  m$coefficients[['dividend']] = sumcoefs
  robust_se[['dividend']] <- se
  robust_pvalue[['dividend']] <- pvalue
  robust_tstat[['dividend']] <- tstat
  # Sum of all 4 coefficients
  sumcoefs <- m$coefficients[['CO']] + m$coefficients[['CO_young']] + m$coefficients[['CO_underinvest']] + m$coefficients[['CO_underinvest_young']]
  lh <- linearHypothesis(m, c("CO + CO_young + CO_underinvest + CO_underinvest_young = 0"), vcov.=vcov)
  tstat <- sqrt(lh$F[2])
  se <- abs(sumcoefs / tstat)
  pvalue <- 2*pt(-abs(tstat), df=length(dataie)-1)
  m$coefficients[['opercycle']] = sumcoefs
  robust_se[['opercycle']] <- se
  robust_pvalue[['opercycle']] <- pvalue
  robust_tstat[['opercycle']] <- tstat
  # Add to the list of results
  models[['table3YOUNG']][[cscore_name]] <- m
  models_se[['table3YOUNG']][[cscore_name]] <- robust_se
  models_pvalue[['table3YOUNG']][[cscore_name]] <- robust_pvalue
  models_tstat[['table3YOUNG']][[cscore_name]] <- robust_tstat
  
}

# Output Table 2
keep = c('CO$', 'CO_underinvest$', 'dividend$')
labels = c('CONS', 'CONS*UnderInvest', 'CONS + CONS*UnderInvest')
addlines = list(c('Controls', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'))

stargazer(models[['table2']], dep.var.caption='$Investment_{t+1}$', dep.var.labels='',
          se=models_tstat[['table2']], p=models_pvalue[['table2']], t=models_tstat[['table2']],
          keep=keep, covariate.labels=labels, column.labels=cscores_names,
          add.lines=addlines,
          keep.stat=c('rsq','n'),
          type='latex', float=F, out=paste0(folder_tables,'LaraEtAl_underinvest.tex'))
          #type='text')

## DEBUG ##
# stargazer(models[['table2']], dep.var.caption='$Investment_{t+1}', dep.var.labels='',
#           se=models_se[['table2']], p=models_pvalue[['table2']], t=models_tstat[['table2']],
#           keep=keep, covariate.labels=labels, column.labels=cscores_names,
#           add.lines=addlines,
#           keep.stat=c('rsq','n'),
#           type='text')


# Output Table 3 - IA
keep = c('CO$', 'CO_IA$', 'dividend$', 'CO_underinvest$', 'CO_underinvest_IA$', 'opercycle$')
labels = c('CONS', 'CONS*IA', 'CONS + CONS*IA', 'CONS*UnderInvest', 'CONS*UnderInvest*IA', 'CONS + CONS*IA + CONS*UnderInvest + CONS*UnderInvest*IA')
addlines = list(c('Controls', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'))

stargazer(models[['table3IA']], dep.var.caption='$Investment_{t+1}$', dep.var.labels='',
          se=models_tstat[['table3IA']], p=models_pvalue[['table3IA']], t=models_tstat[['table3IA']],
          keep=keep, order=keep, covariate.labels=labels, column.labels=cscores_names,
          add.lines=addlines,
          keep.stat=c('rsq','n'),
          type='latex', float=F, out=paste0(folder_tables,'LaraEtAl_IA.tex'))
          #type='text')

## DEBUG ##
# stargazer(models[['table3IA']], dep.var.caption='$Investment_{t+1}', dep.var.labels='',
#           se=models_se[['table3IA']], p=models_pvalue[['table3IA']], t=models_tstat[['table3IA']],
#           keep=keep, order=keep, covariate.labels=labels, column.labels=cscores_names,
#           add.lines=addlines,
#           keep.stat=c('rsq','n'),
#           type='text')

# Output Table 3 - NCR
keep = c('CO$', 'CO_ncr$', 'dividend$', 'CO_underinvest$', 'CO_underinvest_ncr$', 'opercycle$')
labels = c('CONS', 'CONS*NCR', 'CONS + CONS*NCR', 'CONS*UnderInvest', 'CONS*UnderInvest*NCR', 'CONS + CONS*NCR + CONS*UnderInvest + CONS*UnderInvest*NCR')
addlines = list(c('Controls', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'))

stargazer(models[['table3NCR']], dep.var.caption='$Investment_{t+1}$', dep.var.labels='',
          se=models_tstat[['table3NCR']], p=models_pvalue[['table3NCR']], t=models_tstat[['table3NCR']],
          keep=keep, order=keep, covariate.labels=labels, column.labels=cscores_names,
          add.lines=addlines,
          keep.stat=c('rsq','n'),
          type='latex', float=F, out=paste0(folder_tables,'LaraEtAl_NCR.tex'))
          #type='text')

## DEBUG ##
# stargazer(models[['table3NCR']], dep.var.caption='$Investment_{t+1}', dep.var.labels='',
#           se=models_se[['table3NCR']], p=models_pvalue[['table3NCR']], t=models_tstat[['table3NCR']],
#           keep=keep, order=keep, covariate.labels=labels, column.labels=cscores_names,
#           add.lines=addlines,
#           keep.stat=c('rsq','n'),
#           type='text')

# Output Table 3 - YOUNG
keep = c('CO$', 'CO_young$', 'dividend$', 'CO_underinvest$', 'CO_underinvest_young$', 'opercycle$')
labels = c('CONS', 'CONS*Young', 'CONS + CONS*Young', 'CONS*UnderInvest', 'CONS*UnderInvest*Young', 'CONS + CONS*Young + CONS*UnderInvest + CONS*UnderInvest*Young')
addlines = list(c('Controls', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'))

stargazer(models[['table3YOUNG']], dep.var.caption='$Investment_{t+1}$', dep.var.labels='',
          se=models_tstat[['table3YOUNG']], p=models_pvalue[['table3YOUNG']], t=models_tstat[['table3YOUNG']],
          keep=keep, order=keep, covariate.labels=labels, column.labels=cscores_names,
          add.lines=addlines,
          keep.stat=c('rsq','n'),
          type='latex', float=F, out=paste0(folder_tables,'LaraEtAl_YOUNG.tex'))
          #type='text')

## DEBUG ##
# stargazer(models[['table3YOUNG']], dep.var.caption='$Investment_{t+1}', dep.var.labels='',
#           se=models_se[['table3YOUNG']], p=models_pvalue[['table3YOUNG']], t=models_tstat[['table3YOUNG']],
#           keep=keep, order=keep, covariate.labels=labels, column.labels=cscores_names,
#           add.lines=addlines,
#           keep.stat=c('rsq','n'),
#           type='text')


