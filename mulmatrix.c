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
double* C1;
double* C2;
double* C3;

void mulmat(void){     //обычное математическое перемножение матриц
    for (int i = 0; i < AR; i++)
    {
        for (int j = 0; j < BC; j++)
        {
            C1[i*CC+j] = 0;
            for (int k = 0; k < AC; k++)
            {
                C1[i*CC+j] += A[i*AC+k]*B[k*BR+j];
            }
        }
    }
}

void mulstr (void){     //оптимизация через строчную обработку и полусуммы
    for (int i = 0; i < AR; i++){//идём по строкам А и С, i = номер строки в А
        double* c = C2 + i*CC; //указатель на 1 элемент i-ой строки С
        
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

void mulvec(void){      //векторная оптимизация
    memset(C3, 0, sizeof(C3[0])*CR*CC);
    for (int i = 0; i < AR; i++){//идём по строкам А и С, i = номер строки в А
        double* c = C3 + i*CC; //указатель на 1 элемент i-ой строки С
        double* ap = A + i*AC;
        const double* b = B;//указатель на 1 элемент матрицы В
        
        for (int k = AC; k > 0; k--){//AC=BR, идём по столбцам B и элементам в строке А, k = номер столбца А/номер строки В
        
            __m256d a =  _mm256_broadcast_sd (ap);//значение k-го элемента в i-ой строке А множим в вектор
            double* cp = c;
            for (int j = CC>>3; j > 0; j--){//идём по строке С
                _mm256_store_pd(cp, _mm256_fmadd_pd(a, 
                    _mm256_load_pd(b), _mm256_load_pd(cp)));

                b+=4;
                cp+=4;
                _mm256_store_pd(cp, _mm256_fmadd_pd(a, 
                    _mm256_load_pd(b), _mm256_load_pd(cp)));
                b+=4;
                cp+=4;
            }
            ap++;
        }
        
    }

}

int main(int argc, char const *argv[])
{
    A = _mm_malloc(sizeof(double)*AR*AC, 4*(sizeof(double)));
    B = _mm_malloc(sizeof(double)*BR*BC,  4*(sizeof(double)));
    C1 = _mm_malloc(sizeof(double)*CR*CC,  4*(sizeof(double)));
    C2 = _mm_malloc(sizeof(double)*CR*CC,  4*(sizeof(double)));
    C3 = _mm_malloc(sizeof(double)*CR*CC,  4*(sizeof(double)));

    for (int i = 0; i < AC*AR; i++)
    {
        A[i] = drand48()*5; 
    }
    
    for (int i = 0; i <BC*BR; i++)
    {
        B[i] = drand48()*5; 
    }

    struct timespec t0, t1;
    double tres1, tres2, tres3;

    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t0);
        
//    mulmat();

    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);

    tres1 = ((double)(t1.tv_sec - t0.tv_sec)*10e9 + (double)(t1.tv_nsec - t0.tv_nsec))/10e6;
    printf("Время рассчёта матрицы обычным способом: %2.5f mksec \n", tres1);

    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t0);
    
    mulstr();
    
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);

    tres2 = ((double)(t1.tv_sec - t0.tv_sec)*10e9 + (double)(t1.tv_nsec - t0.tv_nsec))/10e6;
    printf("Время рассчёта матрицы СТРОЧНЫМ способом: %2.5f mksec, %.4f  \n", tres2, tres1/tres2);

    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t0);
    
    mulvec();
    
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);

    tres3 = ((double)(t1.tv_sec - t0.tv_sec)*10e9 + (double)(t1.tv_nsec - t0.tv_nsec))/10e6;
    printf("Время рассчёта матрицы ВЕКТОРНЫМ способом: %2.5f mksec, %.4f \n", tres3, tres2/tres3);

    for (int i = 0; i < CR*CC; i++)
            if(C2[i]!=C3[i]) {printf("Результирующие матрицы НЕ равны!\n"); break;}
    
    _mm_free(A);
    _mm_free(B);
    _mm_free(C1);
    _mm_free(C2);
    _mm_free(C3);


    return 0;
}
