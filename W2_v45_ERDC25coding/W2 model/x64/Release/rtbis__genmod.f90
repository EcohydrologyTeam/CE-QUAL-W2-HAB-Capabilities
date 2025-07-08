        !COMPILER-GENERATED INTERFACE MODULE: Wed Jun 25 13:39:16 2025
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE RTBIS__genmod
          INTERFACE 
            RECURSIVE FUNCTION RTBIS(X1,X2,XACC,U,Y,V)
              REAL(KIND=8), INTENT(IN) :: X1
              REAL(KIND=8), INTENT(IN) :: X2
              REAL(KIND=8), INTENT(IN) :: XACC
              REAL(KIND=8), INTENT(IN) :: U
              REAL(KIND=8), INTENT(IN) :: Y
              REAL(KIND=8), INTENT(IN) :: V
              REAL(KIND=8) :: RTBIS
            END FUNCTION RTBIS
          END INTERFACE 
        END MODULE RTBIS__genmod
