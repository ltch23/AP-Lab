#include <stdio.h>
#include <string.h>
#include <mpi.h>
double f(double x){
  return x*x;
}
double Trap(double left_endpt, double right_endpt, int trap_count, double base_len){
  double estimate, x;
  int i;
  estimate = (f(left_endpt) + f(right_endpt))/2.0;
  for(i = 1; i <= trap_count -1; i++){
    x = left_endpt + i*base_len;
    estimate += f(x);
  }
  estimate = estimate*base_len;
  return estimate;
}
int main(int argc, char const *argv[])
{
	int my_rank, com_sx,local_n;
	double local_a,local_b,h,local_int, total_int;
	int n= 100;
	double a = 0;
	double b= 3;

MPI_Init(NULL,NULL);
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
MPI_Comm_size(MPI_COMM_WORLD, &com_sx);

h = (b-a)/n;
local_n = n/com_sx;

local_a = a + my_rank * local_n*h;
local_b = local_a +local_n*h;
local_int = Trap(local_a,local_b,local_n,h);

MPI_Reduce(&local_int , &total_int , 1, MPI_DOUBLE, MPI_SUM, 0,
MPI_COMM_WORLD);

if(my_rank ==0 )
{
	printf("With n = %d trapezoids, our estimate\n", n);
    printf("of the integral from %f to %f = %.15e\n", a, b, total_int);
}


MPI_Finalize();



	return 0;
}
