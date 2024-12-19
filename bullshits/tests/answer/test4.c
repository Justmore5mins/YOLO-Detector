#include <stdio.h>
#include <stdlib.h>

int main(){
    float height, weight;
    float res;

    printf("請輸入身高(cm)");
    scanf("%f",&height);
    height /= 100;
    printf("請輸入體重(kg)");
    scanf(" %f",&weight);
    res = weight/(height*height);
    printf("BMI = %f\n",res);

    printf("--------------------------------");
    return 0;
    system("pause");
}