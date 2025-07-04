SUBROUTINE UPDATE

USE MAIN
USE GLOBAL;     USE NAMESC; USE GEOMC;  USE LOGICC; USE PREC;  USE SURFHE;  USE KINETIC; USE SHADEC; USE EDDY
  USE STRUCTURES; USE TRANS;  USE TVDC;   USE SELWC;  USE GDAYC; USE SCREENC; USE TDGAS;   USE RSTART
  USE MACROPHYTEC; USE POROSITYC; USE ZOOPLANKTONC; USE CEMAVars, ONLY: SD_TC, JDAY_INIT  
  IMPLICIT NONE
  EXTERNAL RESTART_OUTPUT

!***********************************************************************************************************************************
!*                                       Task 2.7: Variable updates for next timestep                                             **
!***********************************************************************************************************************************
    
    SZ     = Z
    SELWS  = ELWS
    SKTI   = KTI
    SBKT   = BKT
    VOLUH2 = QUH1  *DLT
    VOLDH2 = QDH1  *DLT
    TSSUH2 = TSSUH1*DLT
    TSSDH2 = TSSDH1*DLT
    DO JW=1,NWB
     KT = KTWB(JW)
      ELKT(JW) = ELWS(DS(BS(JW)))     !EL(KT,DS(BS(JW)))-Z(DS(BS(JW)))*COSA(BS(JW))
      DO JB=BS(JW),BE(JW)
          
    !** Horizontal diffusivities   ! SW 8/2/2017
      IF(DXI(JW) < 0.0)THEN
      DO I=CUS(JB),DS(JB)-1
        DO K=KT,KBMIN(I)
          DX(K,I) = ABS(U(K,I))*ABS(DXI(JW))
          IF (INTERNAL_WEIR(K,I)) DX(K,I) = 0.0
        END DO
      END DO
      ENDIF
          
 ! CODE MOVED to after wse computation       ELWS(CUS(JB):DS(JB)+1) = EL(KT,CUS(JB):DS(JB)+1)-Z(CUS(JB):DS(JB)+1)*COSA(JB)
        !DO I=US(JB)-1,DS(JB)
        !  AVHR(KT,I) = H1(KT,I)+(H1(KT,I+1)-H1(KT,I))/(0.5D0*(DLX(I)+DLX(I+1)))*0.5D0*DLX(I)                               !SW 07/29/04
        !END DO
        !AVHR(KT,DS(JB)+1)=H1(KT,DS(JB)+1)                                                                              !SW 03/08/05
        DO I=CUS(JB)-1,DS(JB)+1
          DO K=KT,KB(I)     !DO CONCURRENT(K=KT:KB(I))   !FORALL          !DO K=KTWB(JW),KB(I)
            QSS(K,I)   = 0.0D0
            TSS(K,I)   = 0.0D0
            SU(K,I)    = U(K,I)
            SW(K,I)    = W(K,I)
            T2(K,I)    = T1(K,I)
            SAZ(K,I)   = AZ(K,I)
            H2(K,I)    = H1(K,I)
            BHRATIO(K,I) = BH2(K,I)/BH1(K,I)   ! USED FOR TKE COMPUTATION
            BH2(K,I)   = BH1(K,I)
            BHR2(K,I)  = BHR1(K,I)
            AVH2(K,I)  = AVH1(K,I)
            SAVH2(K,I) = AVH2(K,I)
            SAVHR(K,I) = AVHR(K,I)
          END DO                     
        END DO
      END DO
    END DO
    IF (CONSTITUENTS) THEN
      DO JW=1,NWB
        KT = KTWB(JW)
        DO JB=BS(JW),BE(JW)
          DO JC=1,NAC
              DO K=KT,MAX(KB(DS(JB)),KB(US(JB)))
                  CSSUH2(K,CN(JC),JB) = CSSUH1(K,CN(JC),JB)*DLT            ! SW CODE SPEEDUP 6/15/13
                  CSSDH2(K,CN(JC),JB) = CSSDH1(K,CN(JC),JB)*DLT
              ENDDO
          ENDDO
        ENDDO
      ENDDO
      DO JW=1,NWB
        KT = KTWB(JW)
        DO JB=BS(JW),BE(JW)
          DO JC=1,NAC
            DO I=US(JB)-1,DS(JB)+1
              DO K=KT,KB(I)
                DO JE=1,NEP
                  IF (EPIPHYTON_CALC(JW,JE)) EPD(K,I,JE) = DMAX1(EPD(K,I,JE),0.0D0)
                END DO

                IF (SEDIMENT_CALC(JW))THEN
                  SED(K,I) = MAX(SED(K,I),0.0)
                  SEDP(K,I) = DMAX1(SEDP(K,I),0.0D0)
                  SEDN(K,I) = DMAX1(SEDN(K,I),0.0D0)
                  SEDC(K,I) = DMAX1(SEDC(K,I),0.0D0)
                END IF

                CSSB(K,I,CN(JC)) = 0.0D0
                C1S(K,I,CN(JC))  = C1(K,I,CN(JC))
                C2(K,I,CN(JC))   = DMAX1(C1(K,I,CN(JC)),0.0D0)

                DO J=1,NAL                                               !> sch 25Jan2025. Low DO - high mortality option. Loop through the algal state variables.
                  IF(ALG(K,I,J)<ALG_MIN) ALG(K,I,J) = ALG_MIN            !> sch 25Jan2025. Low DO - high mortality option. Ensure a nominal standing crop of algae (ALG_MIN) is maintained.
                END DO                                                   !> sch 25Jan2025. Low DO - high mortality option. End loop through the algal state variables.
				
              END DO
            END DO
          END DO
        END DO
      END DO
    END IF

    !DO JW = 1,NWB
    !  KT = KTWB(JW)
    !  DO M=1,NMC
    !    IF (MACROPHYTE_CALC(JW,M)) THEN
    !      DO JB=BS(JW),BE(JW)
    !        DO I=US(JB),DS(JB)
    !          DO K=KT,KB(I)
    !            SMAC(K,I,M)=MAC(K,I,M)
    !            DO J=1,KMX
    !              SMACRC(J,K,I,M)=MACRC(J,K,I,M)
    !              SMACRM(J,K,I,M)=MACRM(J,K,I,M)
    !            END DO
    !          END DO
    !        END DO
    !      END DO
    !    END IF
    !  END DO
    !END DO

  DO JW = 1,NWB 
    IF (ULTIMATE(JW)) THEN   ! SR 5/15/06
      IF(LAYERCHANGE(JW) == .TRUE.)THEN
      DO K=KTWB(JW),KMX    ! only need to update this for KT - if layer change then update for all variables to be safe esp for seg additions                       !DO K=2,KMX   KTWB(JW),KMX
      RATZ(K,JW)  =  AVH2(K-1,DS(BE(JW)))/AVH2(K,DS(BE(JW)))                                         ! SW 5/20/05
      CURZ1(K,JW) =  2.0D0*H(K,JW)*H(K,JW)/((AVH2(K-1,DS(BE(JW)))+AVH2(K,DS(BE(JW))))*AVH2(K-1,DS(BE(JW))))   ! SW 5/20/05   4/20/16 SPEED
      CURZ2(K,JW) = -2.0D0*H(K,JW)*H(K,JW)/(AVH2(K-1,DS(BE(JW)))*AVH2(K,DS(BE(JW))))                          ! SW 5/20/05
      CURZ3(K,JW) =  2.0D0*H(K,JW)*H(K,JW)/((AVH2(K-1,DS(BE(JW)))+AVH2(K,DS(BE(JW))))*AVH2(K,DS(BE(JW))))     ! SW 5/20/05
      END DO
      ELSE
      DO K=KTWB(JW),KTWB(JW)+1
      RATZ(K,JW)  =  AVH2(K-1,DS(BE(JW)))/AVH2(K,DS(BE(JW)))                                         ! SW 5/20/05
      CURZ1(K,JW) =  2.0D0*H(K,JW)*H(K,JW)/((AVH2(K-1,DS(BE(JW)))+AVH2(K,DS(BE(JW))))*AVH2(K-1,DS(BE(JW))))   ! SW 5/20/05  4/20/16 SPEED
      CURZ2(K,JW) = -2.0D0*H(K,JW)*H(K,JW)/(AVH2(K-1,DS(BE(JW)))*AVH2(K,DS(BE(JW))))                        ! SW 5/20/05
      CURZ3(K,JW) =  2.0D0*H(K,JW)*H(K,JW)/((AVH2(K-1,DS(BE(JW)))+AVH2(K,DS(BE(JW))))*AVH2(K,DS(BE(JW))))     ! SW 5/20/05
      END DO
      ENDIF
    END IF
    END DO
    NIT     =  NIT+1
    ELTM    =  ELTM+DLT
    ELTMS   =  ELTMS+DLT
    ELTMF   =  ELTMF+DLT
    JDAY    =  ELTM/DAY
    ELTMJD  =  JDAY-TMSTRT
    END_RUN =  JDAY >= TMEND
    DLT     =  DMAX1(DLTMIN,DLTFF*CURMAX)    ! SW 7/13/2010
    DLT     =  DMIN1(DLT,1.1*DLTS)
    DLTAV   = (ELTM-TMSTRT*DAY)/NIT
    IF (DLT <  MINDLT) THEN
      MINDLT = DLTS
      JDMIN  = JDAY
    END IF
        IF (JDAY >= DLTD(DLTDP+1))THEN
        DLTDP = DLTDP+1
        DLTMAXX=DLTMAX(DLTDP)
        DLTFF=  DLTF(DLTDP)
        ENDIF
     IF(DLTDP < NDLT .AND. DLTINTER == '      ON')THEN              ! SW 7/13/2010
     DLTMAXX=DLTMAX(DLTDP)+((DLTMAX(DLTDP+1)-DLTMAX(DLTDP))/(DLTD(DLTDP+1)-DLTD(DLTDP)))*(JDAY-DLTD(DLTDP))
     DLTFF=DLTF(DLTDP)+((DLTF(DLTDP+1)-DLTF(DLTDP))/(DLTD(DLTDP+1)-DLTD(DLTDP)))*(JDAY-DLTD(DLTDP))
     ENDIF

    IF (DLT  >  DLTMAXX)DLT=DLTMAXX
    CURMAX = DLTMAXX/DLTFF           ! SW 7/13/2010
    
    IF (INT(JDAY) == JDAYNX) THEN
      JDAYG  = JDAYG+1
      JDAYNX = JDAYNX+1
    END IF
    WRITE (GDCH,'(I3)') GDAY
    CALL GREGORIAN_DATE
    IF(CONSTITUENTS.AND.YEAR/=YEAROLD.AND.CO2YEARLYPPM=='      ON')THEN   ! UPDATE PCO2 FOR PH/TIC IF YEAR CHANGES STEP CHANGES
            IF(YEAR<1980)THEN
             !PCO2 = (0.000041392*REAL(YEAR*YEAR*YEAR) - 0.231409975*REAL(YEAR*YEAR) + 430.804190829*REAL(YEAR) - 266735.857433224)*PALT(DS(BE(1)))*1.0E-6      ! PPM CO2 AND ALTITUDE CORRECTION 
             PCO2 = (0.000041392*REAL(YEAR)*real(YEAR)*real(YEAR) - 0.231409975*REAL(YEAR)*real(YEAR) + 430.804190829*REAL(YEAR) - 266735.857433224)*PALT(DS(BE(1)))*1.0E-6  
            ELSE
             !PCO2  = (0.015903*YEAR*YEAR - 61.799598*YEAR + 60357.055057)*PALT(DS(BE(1)))*1.0E-6
              PCO2  = (0.015903*real(YEAR)*real(year) - 61.799598*real(YEAR) + 60357.055057)*PALT(DS(BE(1)))*1.0E-6            ! SW 2/29/2024
            ENDIF
        YEAROLD=YEAR
    ENDIF
    
    UPDATE_KINETICS = .FALSE.
    IF ((CUF > 0 .AND. MOD(NIT,INT(CUF)) == 0) .OR. (CUF <= 0 .AND. JDAY >= NXTMUK)) THEN      !SR 07/03/2023
        UPDATE_KINETICS = .TRUE.
        SD_tc=JDAY-JDAY_INIT   ! SW 10/16/2022 FOR SED DIAG MODEL
        JDAY_INIT=JDAY
        IF (CUF <= 0) NXTMUK = JDAY + ABS(CUF)                  !SR 07/03/2023
    ENDIF

    
    RETURN
    END SUBROUTINE UPDATE
