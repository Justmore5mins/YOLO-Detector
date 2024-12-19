#include <stdio.h>
#include <stdlib.h>

int main(){
    int hundred;
    printf("請輸入百元鈔張數：");
    scanf("%d",&hundred);

    printf("\n");
    printf("可以換到50元： %d 個\n",hundred*2);
    printf("可以換到10元： %d 個\n",hundred*10);
    system("pause");
}
