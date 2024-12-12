#include <stdio.h>

int main(){
    int b,h;
    printf("請輸入底與高");
    scanf("%d %d",&b,&h);
    printf("面積為");
    printf("%.2f",(float)(b*h)/2);
}