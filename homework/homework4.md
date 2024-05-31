# 来泽远 PB21000164

## T1

### a.

$$
\begin{align}
Throughput &= 10 \times 0.8 \times0.85 \times 0.7 \times \frac{32}{4} \times 1.5G \\
&= 57.12 \space GFLOP/s

\end{align}
$$

### b.

#### (1)

$$
57.12 \space GFLOP/s \times2 = 114.24\space GFLOP/s
$$

#### (2)

$$
57.12 \space GFLOP/s \times \frac{15}{10} = 85.68\space GFLOP/s
$$

#### (2)

$$
57.12 \space GFLOP/s \times \frac{0.95}{0.85} = 65.86\space GFLOP/s
$$

## T2

#### (1)

考虑每一次循环。包含4次读，2次写，4次乘，1次加，1次减。

密度为$\frac{4+1+1}{4+2} = 1$。

#### (2)

```
ADDI 	R2, R0, #2400
ADD  	R2, R2, Rcr
ADDI	R1, R0, #8
MOVI2S	VLR, R1
ADDI	R1, R0, #64
ADDI	R3, R0, #64
Loop:LV	V1, Rar
LV		V3, Rbr
MULSV	V5, V1, V3
LV		V2, Rai
LV		V4, Rbi
MULSV	V6,	V2, V4
SUBSV	V5, V5, V6
SV		Rcr, V5
MULSV	V5, V1, V4
MULSV	V6,	V2, V3
ADDSV	V5, V5, V6
SV		Rci, V5
ADD		Rar, Rar, R1
ADD		Rai, Rai, R1
ADD		Rbr, Rbr, R1
ADD		Rbi, Rbi, R1
ADD		Rcr, Rcr, R1
ADD		Rci, Rci, R1
ADDI	R1, R0, #512
MOVI2S	VLR, R3
SUB		R4, R2, Rcr
BNEZ	R4, Loop
```

### (3)

每次处理128个元素，即64个复数。

6次访存，6个Chimes。

每个复数需要周期为$(6\times64 + 15\times6 + 8\times4 + 5\times2)/64 = 7.75 \approx 8$个周期

### (4)

2个chimes。

每个复数需要周期为$(2\times64 + 15\times6 + 8\times4 + 5\times2)/64 = 3.75 \approx 4$个周期
