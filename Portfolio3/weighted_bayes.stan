data {
  int<lower=0> N;
  vector[N] choice; 
  array[N] real <lower = 0, upper = 1> Source1; // Source 1 is the rating scaled between 0.1 and 0.9
  array[N] real <lower = 0, upper = 1> Source2; // Source 2 is the feedback (choice1 + diff) 
}

transformed data {
  array[N] real l_Source1;
  array[N] real l_Source2;
  vector[N] l_choice;
  l_Source1 = logit(Source1);  // logit transformed Source 1 (to be between -inf and inf)
  l_Source2 = logit(Source2);  // logit transformed Source 2 (to be between -inf and inf)
  l_choice = logit(choice/9);  // logit transformed choice divided by 9 
}

parameters {
  real bias;
  real sd;
  // meaningful weights are between 0 and 1 (theory reasons)
  real<lower = 0, upper = 1> w1; 
  real<lower = 0, upper = 1> w2;
}

model {
  target += normal_lpdf(sd | 0.3, 0.1) - normal_lccdf(0|0.3,0.1); // Avoiding negative sd's
  target += normal_lpdf(bias | 0, 1);
  target += normal_lpdf(w1 | 0.5, .2);
  target += normal_lpdf(w2 | 0.5, .2);
  for (n in 1:N){
    target += normal_lpdf(l_choice[n] | bias + w1 *l_Source1[n] + w2 * l_Source2[n], sd);
  }
}

generated quantities{
  array[N] real log_lik;
  real bias_prior;
  real sd_prior;
  real w1_prior;
  real w2_prior;
  bias_prior = normal_rng(0, 1) ;
  sd_prior = normal_rng(0.3, .1);
  w1_prior = normal_rng(0.5, .2);
  w2_prior = normal_rng(0.5, .2);
  for (n in 1:N){
    log_lik[n] = normal_lpdf(l_choice[n] | bias + w1 * l_Source1[n] + w2 * l_Source2[n],sd);
  }
}
