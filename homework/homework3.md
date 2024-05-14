# 来泽远 PB21000164

## T1

### （1）

$$
\begin{align}
CPI &= 1+0.15*(0.1*3+0.9*0.1*4) \\
&= 1.099
\end{align}
$$

### （2）

$$
\begin{align}
CPI &= 1+0.15*2 \\
&= 1.3
\end{align}
$$

前者更快

## T2

单个

```
L.D F2,0(R1) ; (F2) = X(i)
L.D F6,0(R2) ; (F6) = Y(i)
MUL.D F4,F2,F0 ; (F4) = a*X(i)


DADDIU R1,R1,#8 ; increment X index
DADDIU R2,R2,#8 ; increment Y index
DSLTU R3,R1,R4 ; test: continue loop?
ADD.D F6,F4,F6 ; (F6) = a*X(i) + Y(i)



S.D F6,-8(R2) ; Y(i) = a*X(i) + Y(i)
BNEZ R3,foo ; loop if needed
```

展开4次，使用寄存器重命名

```
foo:L.D F2,0(R1) 	; (F2) = X(i)			1
	L.D F6,0(R2) 	; (F6) = Y(i)			1
	MUL.D F2,F2,F0 	; (F2) = a*X(i)			1
	L.D F3,8(R1) 	; (F3) = X(i)			2
	L.D F5,8(R2) 	; (F5) = Y(i)			2
	MUL.D F3,F3,F0 	; (F3) = a*X(i)			2
	L.D F1,16(R1) 	; (F1) = X(i)			3
	L.D F4,16(R2) 	; (F4) = Y(i)			3
	MUL.D F1,F1,F0 	; (F1) = a*X(i)			3
	L.D F7,24(R1) 	; (F7) = X(i)			4
	L.D F8,24(R2) 	; (F8) = Y(i)			4
	MUL.D F7,F7,F0 	; (F7) = a*X(i)			4
	ADD.D F6,F2,F6 	; (F6) = a*X(i) + Y(i)	1
	ADD.D F5,F3,F5 	; (F5) = a*X(i) + Y(i)	2
	ADD.D F4,F1,F4 	; (F4) = a*X(i) + Y(i)	3
	DADDIU R1,R1,#32 ; increment X index	
	DADDIU R2,R2,#32 ; increment Y index
	ADD.D F7,F8,F4 	; (F7) = a*X(i) + Y(i)	4		
	DSLTU R3,R1,R4 	; test: continue loop?	
	S.D F6,-32(R2) 	; Y(i) = a*X(i) + Y(i)	1
	S.D F5,-24(R2) 	; Y(i) = a*X(i) + Y(i)	2
	S.D F4,-16(R2) 	; Y(i) = a*X(i) + Y(i)	3
	S.D F4,-8(R2) 	; Y(i) = a*X(i) + Y(i)	4
	BNEZ R3,foo 	; loop if needed		
```

每个元素需要$(24+1)/4=6.25$个周期

## T3

```
L.D F2,0(R1) ; (F2) = X(i)
MUL.D F4,F2,F0 ; (F4) = a*X(i)
L.D F6,0(R2) ; (F6) = Y(i)
ADD.D F6,F4,F6 ; (F6) = a*X(i) + Y(i)
S.D F6,0(R2) ; Y(i) = a*X(i) + Y(i)
DADDIU R1,R1,#8 ; increment X index
DADDIU R2,R2,#8 ; increment Y index
DSLTU R3,R1,R4 ; test: continue loop?
BNEZ R3,foo ; loop if needed
```

| 迭代 | 指令    | 发射       | 执行/访存（开始-结束） | 写CDB | 注释 |
| ---- | ------- | ---------- | ---------------------- | ----- | ---- |
| 1    | LD1     | 1          | 2-3                    | 4     |      |
| 1    | MUL     | 2          | 4-19 等LD1             | 20    |      |
| 1    | LD2     | 3          | 4-5                    | 6     |      |
| 1    | ADD     | 4          | 20-30 等MUL            | 31    |      |
| 1    | ST      | 5          | 31 等ADD，一步         | 32    |      |
| 1    | DADDIU1 | 6          | 7                      | 8     |      |
| 1    | DADDIU2 | 7          | 8                      | 9     |      |
| 1    | DSLTU   | 8          | 9                      | 10    |      |
| 1    | BNEZ    | 9          | 10                     |       |      |
| 2    | LD1     | 11（BNEZ） | 12-13                  | 14    |      |
| 2    | MUL     | 12         | 14-29 等LD1            | 30    |      |
| 2    | LD2     | 13         | 14-15                  | 16    |      |
| 2    | ADD     | 14         | 30-40 等MUL            | 41    |      |
| 2    | ST      | 15         | 41                     | 42    |      |
| 2    | DADDIU1 | 16         | 17                     | 18    |      |
| 2    | DADDIU2 | 17         | 18                     | 19    |      |
| 2    | DSLTU   | 18         | 19                     | 20    |      |
| 2    | BNEZ    | 19         | 20                     |       |      |
| 3    | LD1     | 21         | 22-23                  | 24    |      |
| 3    | MUL     | 22         | 24-39                  | 40    |      |
| 3    | LD2     | 23         | 24-25                  | 26    |      |
| 3    | ADD     | 24         | 40-55                  | 56    |      |
| 3    | ST      | 25         | 56                     | 57    |      |
| 3    | DADDIU1 | 26         | 27                     | 28    |      |
| 3    | DADDIU2 | 27         | 28                     | 29    |      |
| 3    | DSLTU   | 28         | 29                     | 30    |      |
| 3    | BNEZ    | 29         | 30                     |       |      |

