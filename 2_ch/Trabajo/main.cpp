#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <fstream>

typedef int T;

using namespace std;

void print(vector<vector<T>> & m){
    int row=m.size();
    int col=m[0].size();
    for (int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            cout<<m[i][j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
    
}

void fill(vector<vector<T>> &m){
    int row=m.size();
    int col=m[0].size();
    for (int i=0;i<row;i++)
        for(int j=0;j<col;j++)
            m[i][j]=rand()%10+1;
}
vector<vector<T>>multiply(vector<vector<T>>m1,vector<vector<T>>m2){
    
    int row1=m1.size();
    int col1=m1[0].size();
    int col2=m2[0].size();
    
    vector<vector<T>> m_rpta(row1,vector<T>(col2));
    for (int i=0;i<row1;i++)
        for(int j=0;j<col2;j++)
            for(int k=0;k<col1;k++)
                    m_rpta[i][j]+=m1[i][k]*m2[k][j];    
    return m_rpta;
            
}
vector<vector<T>>mult_blocked (vector<vector<T>>m1,vector<vector<T>>m2){
    
    int n=m1.size(); 
    int block_size=n/10;
    vector<vector<T>> m_rpta(n,vector<T>(n));
    for(int i=0; i<n; i+=block_size )
        for(int j=0; j<n; j+=block_size)
            for(int k=0; k<n; k+=block_size )
                for(int y=i; y<i+block_size; y++ )
                    for(int x=j; x<j+block_size; x++ )
                        for (int z=k; z<k+block_size; z++ )
                            m_rpta[y][x] += m1[y][z] * m2[z][x]; 
    return m_rpta;
}


int main(){
int N,row1,col1,row2,col2;
cout<<"N:"; cin>>N;  

row1=col1=row2=col2=N;
if(col1!=row2) return 0;
vector<vector<T>> m_matr1(row1,vector<T>(col1));
vector<vector<T>> m_matr2(row2, vector<T>(col2));
vector<vector<T>> m_multiply(row1, vector<T>(col2));
vector<vector<T>> m_blocked(row1, vector<T>(col2));
srand(time(NULL));
fill(m_matr1);fill(m_matr2);

//PRIMER ALGORITMO
m_multiply=multiply(m_matr1,m_matr2);
//SEGUNDO ALGORITMO
m_blocked=mult_blocked(m_matr1,m_matr2);
 
return 0;
}
