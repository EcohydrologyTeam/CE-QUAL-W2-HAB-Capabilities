      MODULE BUILDVERSION !
      INTEGER :: INTEL_COMPILER_VERSION= __INTEL_COMPILER
      INTEGER :: INTEL_COMPILER_BUILD_DATE=__INTEL_COMPILER_BUILD_DATE
      CHARACTER(20) :: compiler_build_date=__DATE__
      CHARACTER(20) :: COMPILER_BUILD_TIME=__TIME__
      CHARACTER(40) :: BUILDTIME=__DATE__//' '//__TIME__    ! V4.5 2022.1 Intel compiler Updated  Date
      END MODULE BUILDVERSION