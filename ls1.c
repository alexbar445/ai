#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
pthread_mutex_t mutex1=PTHREAD_MUTEX_INITIALIZER;
void print_list(double* p,int m){
	int n;
	for (n=0;n<m;n++){
		printf("%lf\n",p[n]);
	}
	printf("\n");
}

void copy(double* input,double* output,int m){
	int n;
	for(n=0;n<m;n++){
		output[n]=input[n];
	}
}
void two_copy(double** input,double** output,int one,int two){
	int n,m;
	for (m=0;m<one;m++){
		for (n=0;n<two;n++){
			output[m][n]=input[m][n];
		}
	}
}
void random_list(double* a,int m){
	static int n;
	for (n=0;n<m;n++){
		a[n]=(rand()%2000)/(double)2000-0.5;
	}
}

double* list_mply(double* a,double* b,double* output,int m){
	int n;
	for(n=0;n<m;n++){
		output[n]=a[n]*b[n];
	}
	return output;
}

double* square(double* a,double* output,int m){
	int n;
	for(n=0;n<m;n++){
		output[n]=a[n]*a[n];
	}
	return output;
}

double* list_div(double* a,double* b,double* output,int m){
	int n;
	for(n=0;n<m;n++){
		output[n]=a[n]/b[n];
	}
	return output;
}

void list_div_int(double* a,double b,double* output,int m){
	int n;
	for(n=0;n<m;n++){
		output[n]=a[n]/b;
	}
}

void list_plus(double* a,double* b,double* output,int m){
	int n;
	for (n=0;n<m;n++){
		output[n]=a[n]+b[n];
	}
}

double * list_plus_int(double* a,int b,double* output,int m){
	int n;
	for (n=0;n<m;n++){
		output[n]=a[n]+b;
	}
	return output;
}
	
void list_minus(double* a,double* b,double* output,int m){
	int n;
	for (n=0;n<m;n++){
		output[n]=a[n]-b[n];
	}
}

double add(double* a,int m){
	int n;
	double num=0;
	for (n=0;n<m;n++){
		num+=a[n];
	}
	return num;
}
double* list_log(double* a,double* output,int m){
	int n;
	for (n=0;n<m;n++){
		output[n]=log(a[n]);
	}
	//donot use tow time
	return output;
}
int take_largest(double* a,int m){
	int n;
	int x=0;
	for(n=0;n<m;n++){
		if (a[x]<a[n]){
			x=n;
		}
	}
	return x;
}
//#####################################################################################################
struct ship{
	double*** c_wdf;
	double** c_bdf;
	double** loss;
	int x;
	int y;
};

struct count{
	int* data;
	double speed;
	int blank;
	int run_time;
	int run_part;
};
double list_mply_int(double* a,double b,double* output,int m){
	int n;
	for (n=0;n<m;n++){
		output[n]=a[n]*b;
	}
}

double*** initialization(int a,int b,int c){
	double*** output=calloc(a,sizeof(double**));
	int n,m,x;
	for (n=0;n<a;n++){
		output[n]=calloc(b,sizeof(double*));
		for(m=0;m<b;m++){
			output[n][m]=calloc(c,sizeof(double));
			for (x=0;x<c;x++){
				output[n][m][x]=0;
			}
		}
	}
	return output;
}
void free_list(double*** input,int a,int b){
	int n,m,x;
	for (n=0;n<a;n++){
		for(m=0;m<b;m++){
			free(input[n][m]);
			input[n][m]=NULL;
		}
		free(input[n]);
		input[n]=NULL;
	}
}
void ReLU(double* input,int m){
	int n;
	for (n=0;n<m;n++){
		if (input[n]<0){
			input[n]=0;
		}
	}
}
void Sigmoid(double* input,double* output,int m){
	int n ;
	for (n=0;n<m;n++){
		output[n]=1/(1+exp(-input[n]));
	}
}
void run_n(double** w,double* b,double* input,double* output,int x,int y){
	int n;
	double pp[x];
	for (n=0;n<y;n++){
		output[n]=add(list_mply(w[n],input,pp,x),x)+b[n];
	}
}
void run_c(double** w,double** input,double** output){
	int a=1;
}
void run_df(double** w,double* b,struct ship input,struct count data,int x,int y){
	int n,m,p;
	
	for (n=0;n<y;n++){
		for (m=0;m<x;m++){
			for (p=1;p<data.blank;p++){
				input.c_wdf[0][n][m]+=input.c_wdf[p][n][m];
			}
		}
		for (p=1;p<data.blank;p++){
			input.c_bdf[0][n]+=input.c_bdf[p][n];
		}
	}
	
	for (n=0;n<y;n++){
		list_mply_int(input.c_wdf[0][n],-data.speed/data.blank,input.c_wdf[0][n],x);
		list_plus(w[n],input.c_wdf[0][n],w[n],x);
	}
	
	list_mply_int(input.c_bdf[0],-data.speed/data.blank,input.c_bdf[0],y);
	list_plus(b,input.c_bdf[0],b,y);
}

double entropy_lost(double* answer,double* input,int m){
	int n;
	double cache0[m];
	for (n=0;n<m;n++){
		if (input[n]==0.0||input[n]==1.0){
			if (answer[n]==input[n]){
				cache0[n]=0;
			}
			else{
				cache0[n]=50;
			}
		}
		else{
			cache0[n]=-(answer[n]*log(input[n])+(1-answer[n])*log(1-input[n]));
		}
	}
	return add(cache0,m); 
}

double* get_c_zdf(double* answer,double* input,double* output,int m){
	double cache0[m];
	int n;
	for (n=0;n<m;n++){
		output[n]=input[n]-answer[n];
	}
	return output;
}

double* get_df(double* a,double** w,double* c_zdf,struct ship output,int time,int x,int y){
	int n,m;
	
	for (n=0;n<y;n++){
		list_mply_int(a,c_zdf[n],output.c_wdf[time][n],x);//c_wdf
	} 
	
	copy(c_zdf,output.c_bdf[time],y);//c_bdf
	
	
	double cache0[y][x];//new_loss
	
	double sum[x];
	for (n=0;n<y;n++){
		list_mply_int(w[n],c_zdf[n],cache0[n],x);
		for (m=0;m<x;m++){
			sum[m]+=cache0[n][m];
		}
		
	}
	copy(sum,output.loss[time],x);
}

//#######################################################################################################################################
int main(){
	int n,m,x,oc1;
	int time1,time2;

	

	
	////##############################################################
	double*** picture=initialization(60000,28,28);
	int* train_labels=calloc(60000,sizeof(int));
	FILE *pf = fopen("data.txt","r");
	for (n=0;n<784*60000;n++){
		if (n%78400==0){
			printf("%d %d %d \n",n/784,(n/28)%28,n%28);
		}
		
		fscanf(pf,"%lf",&picture[n/784][(n/28)%28][n%28]);
		picture[n/784][(n/28)%28][n%28]/=255;
	}
	printf("%d %d %d \n",n/784,(n/28)%28,n%28);
	printf("ok");

	pf=fopen("answer.txt","r");
	for (n=0;n<60000;n++){
		if (n%100==0){
			printf("%d\n",n);
		}
		fscanf(pf,"%d",&train_labels[n]);
	}
	printf("%d",n);
	printf("ok\n");
	fclose(pf);
	

	
	/////################################################################test
	//for (n=0;n<60000;n++){
	//	for(m=0;m<784;m++){
	//		train_images[n][m]=0.5;
	//		train_labels[n]=1;
	//	}
	//}
	/////################################################################
	int data[3]={784,256,10};
	struct count c1={data,0.06 ,30,1,30000};
	//struct count c1={data,0.05,50,2,50000};
	
	
	double** train_images=*initialization(1,60000,data[0]);
	for (n=0;n<60000;n++){
		for(m=0;m<784;m++){
			train_images[n][m]=picture[n][m/28][m%28];
		}
	}
	double c_zdf[data[2]];
	double cache2[data[2]];
	double cache3[data[1]];
	
	double** w0=*initialization(1,data[1],data[0]);
	double** w1=*initialization(1,data[2],data[1]);
	
	double** w[2]={w0,w1};
	
	
	double b0[data[1]];
	double b1[data[2]];
	
	double* b[2]={b0,b1};
	

	double*** c_wdf0=initialization(c1.blank,data[1],data[0]);
	double** c_bdf0=*initialization(1,c1.blank,data[1]);
	
	double** loss0=*initialization(1,c1.blank,data[0]);
	
	double*** c_wdf1=initialization(c1.blank,data[2],data[1]);
	double** c_bdf1=*initialization(1,c1.blank,data[2]);
	
	double** loss1=*initialization(1,c1.blank,data[1]);
	
	
	struct ship s1={c_wdf1,c_bdf1,loss1,data[1],data[2]};
	struct ship s0={c_wdf0,c_bdf0,loss0,data[0],data[1]};

	

	random_list(b0,data[1]);
	random_list(b1,data[2]);
	for (n=0;n<data[1];n++){
		random_list(w0[n],data[0]);
	}
	for (n=0;n<data[2];n++){
		random_list(w1[n],data[1]);
	}
	
	struct num{
		int x;
		int time;
	};
	
	double point[c1.blank];
	struct num n1={0,0};
	//###########################################################################
	void* run(void* y){
		int i1,i2;
		int x=*(double*)y;
		double input[data[0]];
		double output1[data[1]];
		double output2[data[2]],output[data[2]];
		double answer[data[2]];
		
		double* io[3]={input,output1,output2};
		double o_zdf[data[1]];
		
		
		double** rw0=*initialization(1,data[1],data[0]);
		double** rw1=*initialization(1,data[2],data[1]);
	
		double** rw[2]={rw0,rw1};

		double rb0[data[1]];
		double rb1[data[2]];
	
		double* rb[2]={rb0,rb1};
		
		
		pthread_mutex_lock(&mutex1);
		
		copy(b[0],rb[0],data[1]);
		copy(b[1],rb[1],data[2]);
		for (i2=0;i2<data[1];i2++){
			copy(w[0][i2],rw[0][i2],data[0]);
		}
		for (i2=0;i2<data[2];i2++){
			copy(w[1][i2],rw[1][i2],data[1]);
		}
		
		pthread_mutex_unlock(&mutex1);
		for(i1=x;i1<x+n1.time;i1++){
			io[0]=train_images[i1];
			
			answer[train_labels[i1]]=1.0;
			
			run_n(rw[0],rb[0],io[0],io[1],data[0],data[1]);
			ReLU(io[1],data[1]);
			run_n(rw[1],rb[1],io[1],io[2],data[1],data[2]);
			Sigmoid(io[2],output,data[2]);
			
			get_c_zdf(answer,output,c_zdf,data[2]);
			
			get_df(io[1],rw[1],c_zdf,s1,i1%c1.blank,data[1],data[2]);
			
			for (i2=0;i2<data[1];i2++){
				if (*(io[1]+i2)==0){
					o_zdf[i2]=0;
				}
				else{
					o_zdf[i2]=1;
				}
			}
			
			list_mply(o_zdf,s1.loss[i1%c1.blank],s1.loss[i1%c1.blank],data[1]);
			
			get_df(io[0],rw[0],s1.loss[i1%c1.blank],s0,i1%c1.blank,data[0],data[1]);
			
			point[i1%c1.blank]=entropy_lost(answer,output,data[2]);

			answer[train_labels[i1]]=0.0;
		}
		free_list(&rw[0],1,data[1]);
		free_list(&rw[1],1,data[2]);
	}
	///########################################################################################################################################################
	int pthread=6;//837
	pthread_t *tid=calloc(pthread,sizeof(pthread_t));
	double a[5];
	double ship[6]={0,5,10,15,20,25};
	printf("%lf",c1.run_time*c1.run_part/c1.blank);
	n1.time=c1.blank/pthread;
	for (m=0;m<c1.run_time*c1.run_part/c1.blank-10;m++){

		list_plus_int(ship,c1.blank,ship,pthread);
		for (n=0;n<pthread;n++){
			pthread_create(&tid[n],NULL,run,&ship[n]);
		}
		
		for (n=0;n<pthread;n++){
			pthread_join(tid[n],NULL);
		}
		//######################################################################## 
		
		
		run_df(w[1],b[1],s1,c1,data[1],data[2]);
		
		run_df(w[0],b[0],s0,c1,data[0],data[1]);
		a[m%5]=add(point,c1.blank)/(double)c1.blank;
		if (m%5==0){
			time2=clock();
	
			printf("\n%lf\n",add(a,5)/5);
				
			printf("%d %d\n",m,m*c1.blank);
			printf("%lf/s\n",(double)c1.blank*CLOCKS_PER_SEC/(time2-time1)*5);
			printf("%lf finish\n\n",(c1.run_time*c1.run_part-m*c1.blank)*(double)(time2-time1)/((double)c1.blank*CLOCKS_PER_SEC)/5);
			time1=clock();
		}
	}
	double answer[10];
	n=0,x=0;
	for(m=55000;m<60000;m++){
		printf(">>%lf",(double)(m-55000)/5000);
		double input[data[0]];
		double output1[data[1]];
		double output2[data[2]],output[data[2]];

		double* io[3]={input,output1,output2};
		double o_zdf[data[1]];
		
		io[0]=train_images[m];
		
		answer[train_labels[m]]=1.0;
		run_n(w[0],b[0],io[0],io[1],data[0],data[1]);
		ReLU(io[1],data[1]);

		run_n(w[1],b[1],io[1],io[2],data[1],data[2]);
		Sigmoid(io[2],output,data[2]);
		answer[train_labels[m]]=0.0;
		if (take_largest(output,data[2])==train_labels[m]){
			x++;
		}
		n++;
		printf("%d\n",n);
	}
	printf("%lf\n",(double)x/n);
	return 0;
}
