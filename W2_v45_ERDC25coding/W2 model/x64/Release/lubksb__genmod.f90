        !COMPILER-GENERATED INTERFACE MODULE: Mon Feb  3 16:58:28 2025
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE LUBKSB__genmod
          INTERFACE 
            RECURSIVE SUBROUTINE LUBKSB(A,N,NP,INDX,B)
              INTEGER(KIND=4) :: NP
              INTEGER(KIND=4) :: N
              REAL(KIND=8) :: A(NP,NP)
              INTEGER(KIND=4) :: INDX(NP)
              REAL(KIND=8) :: B(N)
            END SUBROUTINE LUBKSB
          END INTERFACE 
        END MODULE LUBKSB__genmod
