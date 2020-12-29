#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>

// TO CALCULATE AVERAGE
float compute_avg(float *array, int num_elements)
 {
  float sum = 0.f;
  int i;
  for (i = 0; i < num_elements; i++) {
    sum += array[i];
  }
  return sum / num_elements;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: avg num_elements_per_proc\n");
    exit(1);
  }

  int num_elements_per_proc = atoi(argv[1]);

  MPI_Init(NULL, NULL);

  int world_rank,world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); 
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  float *experience = NULL;
  float *Salary = NULL;
  float exp[]={9,9,13,15,8,6,13,5,15,6,14,3,5,9,14,5,14,14,13,13,8,7,4,13,15,13,12,5,4,3,8,5,4,9,3,5,10,3,9,12,12,6,14,12,7,6,15,12,4,4};
  float salary[]={32646,34043,46039,46790,30346,32498,49915,33129,49876,34799,48182,34579,32096,33596,47256,33759,47134,46207,46765,49019,30394,32728,31897,49098,49190,49002,46755,30422,34779,33567,31720,32810,30277,34274,32275,32170,34959,33768,30330,48682,46075,32353,47932,49652,32924,30380,48870,47663,34849,34849};
  
  if (world_rank == 0) 
  {
    experience = exp;
    Salary = salary;
  }
 
  // BUFFER TO HOLD SUBSET OF DATA
  float *sub_experience = (float *)malloc(sizeof(float) * num_elements_per_proc);
  assert(sub_experience != NULL);
  float *sub_salary = (float *)malloc(sizeof(float) * num_elements_per_proc);
  assert(sub_salary != NULL);
  MPI_Bcast(&experience,1,MPI_CHAR,0,MPI_COMM_WORLD);
  MPI_Bcast(&salary,1,MPI_CHAR,0,MPI_COMM_WORLD);
  // SCATTER
  MPI_Scatter(experience, num_elements_per_proc, MPI_FLOAT, sub_experience,
              num_elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatter(Salary, num_elements_per_proc, MPI_FLOAT, sub_salary,
              num_elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // AVERAGE FOR SCATTERED DATA
  float sub_avg = compute_avg(sub_experience, num_elements_per_proc);
  float sub_avg_1 = compute_avg(sub_salary, num_elements_per_proc);
  
  // GATHER ALL AVERAGE CALCULATED SUBSETS
  float *sub_avgs = NULL;
  if (world_rank == 0) {
    sub_avgs = (float *)malloc(sizeof(float) * world_size);
    assert(sub_avgs != NULL);
  }
  float *sub_avgs_1 = NULL;
  if (world_rank == 0) {
    sub_avgs_1 = (float *)malloc(sizeof(float) * world_size);
    assert(sub_avgs_1 != NULL);
  }
  MPI_Gather(&sub_avg, 1, MPI_FLOAT, sub_avgs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gather(&sub_avg_1, 1, MPI_FLOAT, sub_avgs_1, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (world_rank == 0) 
  {
    //EXPERIENCE
    float avg_exp = compute_avg(sub_avgs, world_size);
    printf("\n****************************************************************");
    printf("\nAverage of EXPERIENCE(scatter and gather): %f\n", avg_exp);
    float original_data_avg =
      compute_avg(experience, num_elements_per_proc * world_size);
    printf("Average of EXPERIENCE (Function): %f\n", original_data_avg);
    //SALARY
    printf("\n****************************************************************");
    float avg_sal = compute_avg(sub_avgs_1, world_size);
    printf("\nAverage of SALARY(scatter and gather): %f\n", avg_sal);
    float original_data_avg_1 =
      compute_avg(Salary, num_elements_per_proc * world_size);
    printf("Average of SALARY (Function): %f\n", original_data_avg_1);
    printf("\n****************************************************************");
    int n = 50, i;
    float top,down,m,c,rmse=0,y_pred;
    float ss_tot=0,ss_res=0,r2;
    for(i=0;i<n;i++)
    {
        top += (exp[i] - avg_exp) * (salary[i] - avg_sal);
        down += pow((exp[i] - avg_exp), 2);
        m = top / down;
        c = avg_sal - (m * avg_exp);
    }
    printf("\nThe coefficients are:\n%f\t%f\n",m,c);
    for(i=0;i<n;i++)
    {
        y_pred = c + m * exp[i];
        rmse += pow((salary[i] - y_pred) , 2);
    }
    printf("\nThe RMSE value is :%f\n",sqrt(rmse/n));
    for(i=0;i<n;i++)
    {
        y_pred = c + m * exp[i];
        ss_tot += pow((salary[i] - avg_sal), 2);
        ss_res += pow((salary[i] - y_pred) ,2);
    }
    r2 = 1 - (ss_res/ss_tot);
    printf("\nR-Square Value :%f\n",r2);
    printf("\n****************************************************************\n");

}
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}