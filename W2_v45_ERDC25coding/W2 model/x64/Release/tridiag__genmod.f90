        !COMPILER-GENERATED INTERFACE MODULE: Mon Feb  3 17:00:16 2025
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE TRIDIAG__genmod
          INTERFACE 
            RECURSIVE SUBROUTINE TRIDIAG(A,V,C,D,S,E,N,U)
              INTEGER(KIND=4), INTENT(IN) :: N
              INTEGER(KIND=4), INTENT(IN) :: E
              REAL(KIND=8), INTENT(IN) :: A(E)
              REAL(KIND=8), INTENT(IN) :: V(E)
              REAL(KIND=8), INTENT(IN) :: C(E)
              REAL(KIND=8), INTENT(IN) :: D(E)
              INTEGER(KIND=4), INTENT(IN) :: S
              REAL(KIND=8), INTENT(OUT) :: U(N)
            END SUBROUTINE TRIDIAG
          END INTERFACE 
        END MODULE TRIDIAG__genmod
