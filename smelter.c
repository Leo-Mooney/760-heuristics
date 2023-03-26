#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include<string.h>


double pots[51][3] = 
    {{99.136, 0.051, 0.497},
    {99.733, 0.064, 0.138},
    {99.755, 0.083, 0.149},
    {99.198, 0.318, 0.206},
    {99.297, 0.284, 0.33},
    {99.23, 0.327, 0.393},
    {99.485, 0.197, 0.156},
    {99.709, 0.011, 0.056},
    {99.729, 0.007, 0.012},
    {99.118, 0.434, 0.377},
    {99.372, 0.01, 0.349},
    {99.505, 0.028, 0.433},
    {99.187, 0.296, 0.335},
    {99.043, 0.224, 0.531},
    {99.206, 0.166, 0.146},
    {99.395, 0.188, 0.328},
    {99.436, 0.199, 0.303},
    {99.796, 0.009, 0.144},
    {99.186, 0.397, 0.065},
    {99.455, 0.079, 0.278},
    {99.553, 0.084, 0.353},
    {99.539, 0.017, 0.201},
    {99.38, 0.082, 0.239},
    {99.504, 0.009, 0.273},
    {99.391, 0.261, 0.297},
    {99.374, 0.015, 0.578},
    {99.462, 0.179, 0.109},
    {99.03, 0.213, 0.459},
    {99.328, 0.131, 0.371},
    {99.674, 0.055, 0.249},
    {99.413, 0.137, 0.1},
    {99.538, 0.046, 0.151},
    {99.41, 0.109, 0.08},
    {99.163, 0.324, 0.343},
    {99.502, 0.036, 0.412},
    {99.66, 0.083, 0.069},
    {99.629, 0.156, 0.069},
    {99.592, 0.171, 0.008},
    {99.684, 0.011, 0.106},
    {99.358, 0.227, 0.137},
    {99.145, 0.161, 0.403},
    {99.729, 0.028, 0.123},
    {99.335, 0.181, 0.351},
    {99.725, 0.094, 0.14},
    {99.124, 0.325, 0.015},
    {99.652, 0.068, 0.029},
    {99.091, 0.268, 0.565},
    {99.426, 0.146, 0.256},
    {99.383, 0.266, 0.039},
    {99.481, 0.147, 0.327},
    {99.163, 0.121, 0.71}};

double grade_min_Al[11] = {95.00,99.10,99.10,99.20,99.25,99.35,99.50,99.65,99.75,99.85,99.90};
double grade_max_Fe[11] = {5.00, 0.81, 0.81, 0.79, 0.76, 0.72, 0.53, 0.50, 0.46, 0.33, 0.30};
double grade_max_Si[11] = {3.00, 0.40, 0.41, 0.43, 0.39, 0.35, 0.28, 0.28, 0.21, 0.15, 0.15};
double grade_value[11] =  {10.00,21.25,26.95,36.25,41.53,44.53,48.71,52.44,57.35,68.21,72.56};

double average(const int x[17][3], const int i, const int e){
  return (pots[x[i][0]][e] + pots[x[i][1]][e] + pots[x[i][2]][e])/3;
}

int max(const int a, const int b, const int c) {
  return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
}

int min(const int a, const int b, const int c) {
  return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
}

double f(const int *crucible, const bool mod) {
  double quality[3];
  double spread[3];
  for(int i=0;i<3;i++){
    quality[i] = (pots[crucible[0]][i] + pots[crucible[1]][i] + pots[crucible[2]][i])/3;
  }
  double tol = 0.000001;
  for(int i=10;i>=0;i--){
    if(quality[0] >= grade_min_Al[i]-tol && quality[1] <= grade_max_Fe[i] + tol && quality[2] <= grade_max_Si[i] + tol) { 
      return grade_value[i];
    } 
  }  
  return 0;
}

double calc_quad(const double p, const double q, const double r) {
    return pow(p - r, 2.0) / pow(q - r, 2.0);
}

double f_mod(const int *crucible){
  double quality[3];
  for(int i=0;i<3;i++){
    quality[i] = (pots[crucible[0]][i] + pots[crucible[1]][i] + pots[crucible[2]][i])/3;
  }
  double tol = 0.000001;
  for(int i=10;i>=0;i--){
    if(quality[0] >= grade_min_Al[i]-tol && quality[1] <= grade_max_Fe[i] + tol && quality[2] <= grade_max_Si[i] + tol) { 
      double quad = 0;
      if(i!=10 && !(quality[0] >= grade_min_Al[i+1]-tol)){
        quad = calc_quad(quality[0], grade_min_Al[i+1], grade_min_Al[i]);
      }
      else if(i!=10 && !(quality[1] <= grade_max_Fe[i+1] + tol)){
        quad = calc_quad(quality[1], grade_max_Fe[i+1], grade_max_Fe[i]);
      }
      else if(i!=10 && !(quality[2] <= grade_max_Si[i+1] + tol)){
        quad = calc_quad(quality[2], grade_max_Si[i+1], grade_max_Si[i]);
      }
      

      return grade_value[i] + quad;
    } 
  }  
  return 0;
}

double f_mod_spread(const int *crucible, const int max_spread){
  double quality[3];
  for(int i=0;i<3;i++){
    quality[i] = (pots[crucible[0]][i] + pots[crucible[1]][i] + pots[crucible[2]][i])/3;
  }
  double tol = 0.000001;
  int spread = max(crucible[0], crucible[1], crucible[2]) - min(crucible[0], crucible[1], crucible[2]);
  double spread_penalty = (spread > max_spread) ? -10*(spread-max_spread) : 0;
  for(int i=10;i>=0;i--){
    if(quality[0] >= grade_min_Al[i]-tol && quality[1] <= grade_max_Fe[i] + tol && quality[2] <= grade_max_Si[i] + tol) { 
      double quad = 0;
      if(i!=10 && !(quality[0] >= grade_min_Al[i+1]-tol)){
        quad = calc_quad(quality[0], grade_min_Al[i+1], grade_min_Al[i]);
      }
      else if(i!=10 && !(quality[1] <= grade_max_Fe[i+1] + tol)){
        quad = calc_quad(quality[1], grade_max_Fe[i+1], grade_max_Fe[i]);
      }
      else if(i!=10 && !(quality[2] <= grade_max_Si[i+1] + tol)){
        quad = calc_quad(quality[2], grade_max_Si[i+1], grade_max_Si[i]);
      }
      

      return grade_value[i] + quad + spread_penalty;
    } 
  }  
  return 0;
}

double f_spread(const int *crucible, const int max_spread) {
  double quality[3];
  for(int i=0;i<3;i++){
    quality[i] = (pots[crucible[0]][i] + pots[crucible[1]][i] + pots[crucible[2]][i])/3;
  }

  double tol = 0.00001;
  int spread = max(crucible[0], crucible[1], crucible[2]) - min(crucible[0], crucible[1], crucible[2]);
  double spread_penalty = (spread > max_spread) ? -10*(spread-max_spread) : 0;
  for(int i=10;i>=0;i--){
    if(quality[0] >= grade_min_Al[i]-tol && quality[1] <= grade_max_Fe[i] + tol && quality[2] <= grade_max_Si[i] + tol) { 
      return grade_value[i] + spread_penalty;
    } 
  }  
  return 0;
}
double calc_obj(const int x[17][3]) {
  double total = 0;
  for(int i=0; i<17; i++){
    total += f(x[i], false);
  }
  return total;
}

double calc_obj_spread(const int x[17][3], const int max_spread) {
  double total = 0;
  for(int i=0; i<17; i++){
    total += f_spread(x[i], max_spread);
  }
  return total;
}

void print_sol(const int x[17][3]) {
  int MxSprd = 0;
  printf("\n");
  for(int i=0;i<17;i++){
    printf(" %d [", i+1);
    for(int j=0;j<3;j++){
      printf("%d,", x[i][j]+1);
    }
    int spread = max(x[i][0], x[i][1], x[i][2]) - min(x[i][0], x[i][1], x[i][2]);
    MxSprd = (spread > MxSprd) ? spread : MxSprd;
    double al = average(x, i, 0);
    double fe = average(x, i, 1);
    double si = average(x, i, 2);
    printf("] %.3f %%Al, %.3f %%Fe, %.3f %%Si, $%.2f, spread=%d", al, fe, si, f(x[i], false), spread);
    printf("\n");
  }
  printf("Sum = $%.3f", calc_obj(x));
}

void print_sol_spread(const int x[17][3], const int max_spread) {
  int MxSprd = 0;
  for(int i=0;i<17;i++){
    printf(" %d [", i+1);
    for(int j=0;j<3;j++){
      printf("%d,", x[i][j]+1);
    }
    int spread = max(x[i][0], x[i][1], x[i][2]) - min(x[i][0], x[i][1], x[i][2]);
    MxSprd = (spread > MxSprd) ? spread : MxSprd;
    double al = average(x, i, 0);
    double fe = average(x, i, 1);
    double si = average(x, i, 2);
    printf("] %.3f %%Al, %.3f %%Fe, %.3f %%Si, $%.2f, spread=%d", al, fe, si, f_spread(x[i], spread), spread);
    printf("\n");
  }
  printf("Sum = $%.3f, MxSprd = %d", calc_obj_spread(x, max_spread), MxSprd);
}

void next_ascent(int x[17][3]){
  double last_crucible_values[17];
  for(int i=0;i<17;i++){
    last_crucible_values[i] = f_mod(x[i]);
  }
  int o_k = -1;
  int o_l = -1;
  int o_m = -1;
  int o_n = -1;

  while(1){
    for(int k=0;k<16;k++){
      for(int m=0;m<3;m++){
        for(int l=k+1;l<17;l++){
          for(int n=0;n<3;n++){
            if(k==o_k && m==o_m && l==o_l && n==o_n){
              return;
            }
            double swap = x[k][m];
            x[k][m] = x[l][n];
            x[l][n] = swap;
            
            double f_k = f_mod(x[k]);
            double f_l = f_mod(x[l]);

            double delta = f_k + f_l - last_crucible_values[k] - last_crucible_values[l];
            if(delta>0.00001){
              last_crucible_values[k] = f_k;
              last_crucible_values[l] = f_l;
              o_k = k;
              o_m = m;
              o_l = l;
              o_n = n;
            }
            else{
              x[l][n] = x[k][m];
              x[k][m] = swap;
            }
          }
        }
      }
    }
  }
}


void next_ascent_spread(int x[17][3], const int max_spread){
  double last_crucible_values[17];
  for(int i=0;i<17;i++){
    last_crucible_values[i] = f_mod_spread(x[i], max_spread);
  }
  int o_k = -1;
  int o_l = -1;
  int o_m = -1;
  int o_n = -1;

  while(1){
    for(int k=0;k<16;k++){
      for(int m=0;m<3;m++){
        for(int l=k+1;l<17;l++){
          for(int n=0;n<3;n++){
            if(k==o_k && m==o_m && l==o_l && n==o_n){
              return;
            }
            double swap = x[k][m];
            x[k][m] = x[l][n];
            x[l][n] = swap;
            
            double f_k = f_mod_spread(x[k], max_spread);
            double f_l = f_mod_spread(x[l], max_spread);

            double delta = f_k + f_l - last_crucible_values[k] - last_crucible_values[l];
            if(delta>0.001){
              last_crucible_values[k] = f_k;
              last_crucible_values[l] = f_l;
              o_k = k;
              o_m = m;
              o_l = l;
              o_n = n;
            }
            else{
              x[l][n] = x[k][m];
              x[k][m] = swap;
            }
          }
        }
      }
    }
  }
}

void simulated_annealing(int x[17][3], double c1, double alpha, int ck_update_rate) {
  double ck = c1;
  double best = 0;
  unsigned long iter = 0;
  unsigned long accepted = 0;
  int delta_over_0 = 0;
  int last_accepted = 100000;

  double last_crucible_values[17];
  for(int i=0;i<17;i++){
    last_crucible_values[i] = f_mod(x[i]);
  }
  srand(rand());
  while(ck>0.0000000001){
    int k = rand() % 17;
    int l = rand() % 17;
    int m = rand() % 3;
    int n = rand() % 3;
    while(l==k){
      l = rand() % 17;
    }

    double swap = x[k][m];
    x[k][m] = x[l][n];
    x[l][n] = swap;
    
    double f_k = f_mod(x[k]);
    double f_l = f_mod(x[l]);

    double delta = f_k + f_l - last_crucible_values[k] - last_crucible_values[l];
    if(delta>0.0001){
      delta_over_0+=1;
    } 

    if(delta>0.001 || (double)rand() / (double)RAND_MAX < exp(delta/ck)){
      // double obj = calc_obj(x);
      // best = obj > best ? obj : best; 
      last_crucible_values[k] = f_k;
      last_crucible_values[l] = f_l;
      accepted++;
    }
    else {
      // swap back
      x[l][n] = x[k][m];
      x[k][m] = swap;
    }
    
    if(iter%10000000 == 0 && iter != 0){
      // printf("%f, %f, %f, %f, %f", f_k, f_l, last_crucible_values[k], last_crucible_values[l], delta);
      // printf("Iterations: %lu\n", iter);
      printf("Accepted rate (\%): %f\n", 100*(double)accepted/(double)10000000); 
      printf("Ck: %0.15f\n", ck); 
      printf("Obj: %f\n", calc_obj(x));
      printf("Iter: %lu\n", iter);
      // printf("Delta Over 0: %d\n", delta_over_0);
      last_accepted = accepted;
      accepted=0;
    }
    
    if(iter%ck_update_rate==0){
      ck = ck*alpha;
    }
    iter++; 
  }
  // printf("BEST: %f", best);
}

void simulated_annealing_spread(int x[17][3], double c1, double alpha, int ck_update_rate, const int max_spread) {
  double ck = c1;
  double best = 0;
  unsigned long iter = 0;
  unsigned long accepted = 0;
  int delta_over_0 = 0;
  int last_accepted = 100000;

  double last_crucible_values[17];
  for(int i=0;i<17;i++){
    last_crucible_values[i] = f_spread(x[i], max_spread);
  }
  srand(rand());
  while(ck>0.0001){
    int k = rand() % 17;
    int l = rand() % 17;
    int m = rand() % 3;
    int n = rand() % 3;
    while(l==k){
      l = rand() % 17;
    }

    double swap = x[k][m];
    x[k][m] = x[l][n];
    x[l][n] = swap;
    
    double f_k = f_spread(x[k], max_spread);
    double f_l = f_spread(x[l], max_spread);

    double delta = f_k + f_l - last_crucible_values[k] - last_crucible_values[l];
    if(delta>0.0001){
      delta_over_0+=1;
    } 

    if(delta>0.001 || (double)rand() / (double)RAND_MAX < exp(delta/ck)){
      // double obj = calc_obj(x);
      // best = obj > best ? obj : best; 
      last_crucible_values[k] = f_k;
      last_crucible_values[l] = f_l;
      accepted++;
    }
    else {
      // swap back
      x[l][n] = x[k][m];
      x[k][m] = swap;
    }
    
    if(iter%10000000 == 0 && iter != 0){
      // printf("%f, %f, %f, %f, %f", f_k, f_l, last_crucible_values[k], last_crucible_values[l], delta);
      // printf("Iterations: %lu\n", iter);
      printf("Accepted rate (\%): %f\n", 100*(double)accepted/(double)10000000); 
      printf("Ck: %0.15f\n", ck); 
      printf("Obj: %f\n", calc_obj_spread(x, max_spread));
      printf("Iter: %lu\n", iter);
      // printf("Delta Over 0: %d\n", delta_over_0);
      last_accepted = accepted;
      accepted=0;
    }
    
    if(iter%ck_update_rate==0){
      ck = ck*alpha;
    }
    iter++; 
  }
  // printf("BEST: %f", best);
}

void generate_random_x(int x[17][3]) {
  int x_flat[51];
  for(int i=0;i<51;i++){
    x_flat[i] = i;
  }
  srand(rand());
  for(int i=0;i<51;i++){
    int j = rand() % 51;
    int swap = x_flat[i];
    x_flat[i] = x_flat[j];
    x_flat[j] = swap;
  }
  int k = 0;
  for(int i=0;i<17;i++){
    for(int j=0;j<3;j++){
      x[i][j] = x_flat[k];
      k++;
    }
  }
}

int main() {
  srand(time(NULL));
  int x[17][3];

  int max_spread = 8;
  double best_obj = 0;
  int best_x[17][3];
  for(int i=0;i<500000; i++){
    generate_random_x(x);
    next_ascent_spread(x, max_spread);
    double obj = calc_obj_spread(x, max_spread);
    if(obj > best_obj){
      best_obj = obj > best_obj ? obj : best_obj;
      memcpy(best_x, x, sizeof(best_x));
    }
    if(i%1000==0){
      printf("%f : %f\n", best_obj, (double)i/(double)500000);
    }
  }
  print_sol_spread(best_x, max_spread);
  printf("%f", best_obj);
  //
  //
  // double best_obj = 0;
  // int best_x[17][3];
  // for(int i=0;i<500000; i++){
  //   generate_random_x(x);
  //   next_ascent(x);
  //   double obj = calc_obj(x);
  //   if(obj > best_obj){
  //     best_obj = obj > best_obj ? obj : best_obj;
  //     memcpy(best_x, x, sizeof(best_x));
  //   }
  //   if(i%1000==0){
  //     printf("%f : %f\n", best_obj, (double)i/(double)500000);
  //   }
  // }
  // print_sol(best_x);
  // printf("%f", best_obj);
  //
  // generate_random_x(x);
  // simulated_annealing(x, 100, 0.99999999, 1);
  // simulated_annealing_spread(x, 100, 0.999999999, 1, max_spread);
  // print_sol(x);
  return 0;
}

