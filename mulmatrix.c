#include <stdio.h>
#include <time.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>


#define AC 1024
#define AR 1024
#define BC 1024
#define BR AC
#define CC BC
#define CR AR

double* A;
double* B;
double* C;

void mulmat(void){
    for (int i = 0; i < AR; i++)
    {
        for (int j = 0; j < BC; j++)
        {
            C[i*CC+j] = 0;
            for (int k = 0; k < AC; k++)
            {
                C[i*CC+j] += A[i*AC+k]*B[k*BR+j];
            }
        }
    }
}

void mulstr (void){
    for (int i = 0; i < AR; i++){//идём по строкам А и С, i = номер строки в А
        double* c = C + i*CC; //указатель на 1 элемент i-ой строки С
        
        for (int j = 0; j < CC; j++)//идём по (столбцам) элементам в строке C, j = номер столбца в C
            c[j] = 0; //обнуляем элементы строки в матрице С
            
        for (int k = 0; k < AC; k++){//AC=BR, идём по столбцам B и элементам в строке А, k = номер столбца А/номер строки В
            const double* b = B + k*BC;//указатель на 1 элемент строки В
            double a = A[i*AC + k];//значение k-го элемента в i-ой строке  А

            for (int j = 0; j < CC; j++){//идём по строке С
            c[j] += a * b[j];//формируем полусуммы
            }
        }
        
    }
}

void mulvec(void){
    memset(C, 0, sizeof(C[0])*CR*CC);
    for (int i = 0; i < AR; i++){//идём по строкам А и С, i = номер строки в А
        double* c = C + i*CC; //указатель на 1 элемент i-ой строки С
        
        // for (int j = 0; j < CC; j+=4)//идём по (столбцам) элементам в строке C, j = номер столбца в C
        //     _mm256_storeu_pd(c + j + 0, _mm256_setzero_pd()); //обнуляем элементы строки в матрице С
            
        for (int k = 0; k < AC; k++){//AC=BR, идём по столбцам B и элементам в строке А, k = номер столбца А/номер строки В
            const double* b = B + k*BC;//указатель на 1 элемент строки В

            __m256d a =  _mm256_broadcast_sd (A + i*AC + k);//значение k-го элемента в i-ой строке А множим в вектор

            for (int j = 0; j < CC; j+=8){//идём по строке С
                _mm256_storeu_pd(c + j + 0, _mm256_fmadd_pd(a, 
                    _mm256_loadu_pd(b + j + 0), _mm256_loadu_pd(c + j + 0)));
                _mm256_storeu_pd(c + j + 4, _mm256_fmadd_pd(a, 
                    _mm256_loadu_pd(b + j + 4), _mm256_loadu_pd(c + j + 4)));
            //c[j] += a * b[j];//формируем полусуммы
            }
        }
        
    }

}

int main(int argc, char const *argv[])
{
    A = malloc(sizeof(double)*AR*AC);
    B = malloc(sizeof(double)*BR*BC);
    C = malloc(sizeof(double)*CR*CC);

    for (int i = 0; i < AC*AR; i++)
    {
        A[i] = drand48()*5; 
    }
    
    for (int i = 0; i <BC*BR; i++)
    {
        B[i] = drand48()*5; 
    }
    //обычное математическое перемножение матриц
    struct timespec t0, t1;
    float tres1, tres2, tres3;

    clock_gettime(CLOCK_REALTIME, &t0);
        
    mulmat();

    clock_gettime(CLOCK_REALTIME, &t1);

    tres1 = ((t1.tv_sec - t0.tv_sec)*10e9 + (t1.tv_nsec - t0.tv_nsec))/10e6;
    printf("Время рассчёта матрицы обычным способом: %2.5f mksec \n", tres1);

    clock_gettime(CLOCK_REALTIME, &t0);
    
    mulstr();
    
    clock_gettime(CLOCK_REALTIME, &t1);

    tres2 = ((t1.tv_sec - t0.tv_sec)*10e9 + (t1.tv_nsec - t0.tv_nsec))/10e6;
    printf("Время рассчёта матрицы СТРОЧНЫМ способом: %2.5f mksec \n", tres2);

    clock_gettime(CLOCK_REALTIME, &t0);
    
    mulvec();
    
    clock_gettime(CLOCK_REALTIME, &t1);

    tres3 = ((t1.tv_sec - t0.tv_sec)*10e9 + (t1.tv_nsec - t0.tv_nsec))/10e6;
    printf("Время рассчёта матрицы ВЕКТОРНЫМ способом: %2.5f mksec \n", tres3);



    return 0;
}
