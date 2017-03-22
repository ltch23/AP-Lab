#include <iostream>
#include <vector>
#include <stdlib.h>
typedef int T;

using namespace std;


void print(vector<vector<T>> & m){
    int len=m[0].size();
    for (int i=0;i<len;i++){
        for(int j=0;j<len;j++){
            cout<<m[i][j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;

    
}

void fill(vector<vector<T>> &m){
    int len=m[0].size();
    for (int i=0;i<len;i++)
        for(int j=0;j<len;j++)
            m[i][j]=rand()%10+1;
}
        
    

vector<vector<T>>multiply(vector<vector<T>>m1,vector<vector<T>>m2){
    int len=m1[0].size();
    vector<vector<T>> m_rpta(len,vector<T>(len));
    for (int i=0;i<len;i++)
        for(int j=0;j<len;j++)
            for(int k=0;k<len;k++)
                    m_rpta[i][j]+=m1[i][k]*m2[k][j];    
    return m_rpta;
            
}




int main(){
    T len;
cout<<"len:";
cin>>len;    
vector<vector<T>> m_matr1(len,vector<T>(len));
vector<vector<T>> m_matr2(len, vector<T>(len));
vector<vector<T>> m_multiply(len, vector<T>(len));

srand(time(NULL));
fill(m_matr1);
fill(m_matr2);
// print(m_matr1);
// print(m_matr2);
m_multiply=multiply(m_matr1,m_matr2);
// print(m_multiply);
 
return 0;
}
