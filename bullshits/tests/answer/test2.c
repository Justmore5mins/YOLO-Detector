#include <stdio.h>
#include <stdlib.h>

int main(){
    int up, base, height;
    float res;
    printf("請輸入梯形上底長：");
    scanf("%d",&up);
    printf("請輸入梯形下底長：");
    scanf(" %d",&base);
    printf("請輸入梯形的高：");
    scanf(" %d",&height);
    res = (float)(up+base)*height/2;
    printf("梯形的面積是%.1f。",res);
    system("pause");
    return 0;
}
