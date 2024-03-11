// 
data {
  int<lower=1> trials; // lower bound at 1 such that we can't have less than 1 trials
  array[trials] int memory_output; // setting an array of the same length of trials. The choice needs to be 0 or 1 - int for bernoulli (specific case of binomial)
  array[trials] int noisy_choise; // the actual choise after a noisy input 
}

// 
parameters {
  real weight;  // weight is the noise parameter - a probability between 0 and 1 but unbound because we are working in log odds
}

// 
model {

  target += normal_lpdf(weight | 0, 1); // prior for weight is a normal distribution with a mean of 0 and sd of 1 - in logit space
  

  // target is an internal value that the sampler uses to decide where it is in the posterior space and where to move
  // choices are drawn from a logit bernoulli distribution with a rate of the bias (weight * memory_output)
  
for (i in 1:trials)
    target += bernoulli_logit_lpmf(noisy_choise[i] | weight * memory_output[i]); // here, we say: our choice is drawn from a bernoulli distribution. 
    // the actual choice is drawn given the weight and memory_output. In this way, the model can retrieve the informaiton from the weight prior.

}
  // RICCARDO's EXPLANATION OF TARGET +=
    // target indicate how likely,  the two things summed up (log-posterior-prob), given the data
    // These two things are summed up together - summed up at each step andnot increasingly
    // any given slice in time - how are you calculating the log-probability
    
    
generated quantities {
  
   // theta posterior parameter, on a prob scale (0-1).
  real weight_logit_prior; 
  array[trials] int prior_predictive;
  array[trials] int post_predictive; 
  
  
 weight_logit_prior = normal_rng(0,1);
  // converting the posterior estimate from log odds to prob.
for (i in 1:trials) {
  prior_predictive[i] = bernoulli_logit_rng(weight_logit_prior * memory_output[i]);
}
  
    
for (i in 1:trials) {
  post_predictive[i] = bernoulli_logit_rng(weight * memory_output[i]);
}
}