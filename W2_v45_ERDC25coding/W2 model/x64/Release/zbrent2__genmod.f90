        !COMPILER-GENERATED INTERFACE MODULE: Mon Feb  3 16:58:28 2025
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE ZBRENT2__genmod
          INTERFACE 
            RECURSIVE FUNCTION ZBRENT2(FUNC,BARG)
              REAL(KIND=8) :: FUNC
              EXTERNAL FUNC
              REAL(KIND=8) :: BARG
              REAL(KIND=8) :: ZBRENT2
            END FUNCTION ZBRENT2
          END INTERFACE 
        END MODULE ZBRENT2__genmod
