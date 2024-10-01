from sympy

def pid(Kp:float, Ki:float, Kd:float,e:object, t:float,τ:float):
    '''
    Kp, Ki, Kd: 比例增益,積分增益,微分增益(const)
    e:誤差= set-now
    t=目前時間
    τ=積分變數
    '''
    return Kp*e(t) + Ki*integrate(e(τ),t) + Kd*(e(t)-e(τ))