# Generating the speed, pressure, sound, and temperature for Machine 2
# ----------------------------------------------------------------------------------------------------------------------------------------------------

# 10752 (historical data) + 4*112 (future data) = 11200

set.seed(456);

OUTPUT <- c(1:11200)
PRESSURE <- c(1:11200)
SPEED <- c(1:11200)
TEMPERATURE <- c(1:11200)
SOUND <- c(1:11200)
prob <- c(1:11200)
VIBRATION <- c(1:11200)
PERIOD <- c(1:11200)

PRESSURE[[1]] <- 100
SPEED[[1]] <- 100
TEMPERATURE[[1]] <- 80
SOUND[[1]] <- 60
VIBRATION[[1]] <- 120


for(period in 1:11200){
  PERIOD[[period]] <- period
  
  if (period > 1){
    if (OUTPUT[[period - 1]] == 'Failure'){
      PRESSURE[[period]] <- 100
      SPEED[[period]] <- 100
      TEMPERATURE[[period]] <- 80
      SOUND[[period]] <- 60
      VIBRATION[[period]] <- 120
    } else{
      PRESSURE[[period]] <- round(PRESSURE[[period - 1]] + rexp(1, rate = 3), digits = 2);
      SPEED[[period]] <- round(SPEED[[period - 1]] - rexp(1, rate = 4), digits = 2);
      TEMPERATURE[[period]] <- round(TEMPERATURE[[period - 1]] + rexp(1, rate = 2), digits = 2);
      SOUND[[period]] <- round(SOUND[[period - 1]] + rexp(1, rate = 0.9), digits = 2);
      VIBRATION[[period]] <- round(VIBRATION[[period - 1]] + rexp(1, rate = 0.4), digits = 2);
    }
  }  
    
  if (PRESSURE[[period]] > 110 & SPEED[[period]] < 95 & TEMPERATURE[[period]] > 125){
    OUTPUT[[period]] <- 'Failure'
  } else{
    if (PRESSURE[[period]] <= 110 & SPEED[[period]] < 95 & TEMPERATURE[[period]] > 125){
      OUTPUT[[period]] <- 'Failure'
    } else{
      if (PRESSURE[[period]] > 110 & SPEED[[period]] >= 95 & TEMPERATURE[[period]] > 125){
        OUTPUT[[period]] <- 'Failure'
      } else{
        if (PRESSURE[[period]] > 110 & SPEED[[period]] < 95 & TEMPERATURE[[period]] <= 125){
          OUTPUT[[period]] <- 'Failure'
        } else{
          if (PRESSURE[[period]] > 110 & SPEED[[period]] >= 95 & TEMPERATURE[[period]] <= 125){
            OUTPUT[[period]] <- 'Failure'
          } else{
            if (PRESSURE[[period]] <= 110 & SPEED[[period]] < 95 & TEMPERATURE[[period]] <= 125){
              OUTPUT[[period]] <- 'Failure'
            } else{
              if (PRESSURE[[period]] <= 110 & SPEED[[period]] >= 95 & TEMPERATURE[[period]] > 125){
                OUTPUT[[period]] <- 'Failure'
              } else{
                p = rbinom(1, 1, 0.01)
                  
                if(p == 1){
                  OUTPUT[[period]] <- 'Failure'
                } else{
                  OUTPUT[[period]] <- 'Normal'
                }
              }
            }
          }
        }
      }  
    }
  }
}	

df <- data.frame('Period'= PERIOD, 'Failures'= OUTPUT, 'Pressure' = PRESSURE, 'Speed' = SPEED, 'Temperature' = TEMPERATURE, 'Sound' = SOUND, 'Vibration' = VIBRATION)

write.csv(df,"./DATASET_MACHINE_2.csv", row.names = FALSE)


