! Sample fortran extension
! 1D advection equation simulator
! Using OpenMP for parallelization

! 移流計算1ステップ
module sample_1d_advection_step_module
    implicit none
    
contains
    ! モジュール内のサブルーチン定義
    subroutine advec_upwind_step(nx, dt, dx, u, q, q_new)
        implicit none

        ! 引数の属性設定
        real, intent(in) :: dt, dx, u
        integer, intent(in) :: nx
        real, intent(in) :: q(nx)
        real, intent(out) :: q_new(nx)

        ! ユーティリティ変数宣言
        integer :: i
        real :: c

        ! CFLnumの計算
        c = u * (dt / dx)

        ! 移流計算(MP並列化) 境界条件: 周期境界, 差分スキーム: 一次風上差分
        if (u >= 0.0) then
            q_new(1) = q(1) - c * (q(1) - q(nx))
            !$omp parallel
            !$omp do private(i)
            do i = 2, nx
                q_new(i) = q(i) - c * (q(i) - q(i - 1))
            end do
            !$omp enddo
            !$omp end parallel
        else
            q_new(nx) = q(nx) - c * (q(1) - q(nx))
            !$omp parallel
            !$omp do private(i)
            do i = 1, nx - 1
                q_new(i) = q(i) - c * (q(i + 1) - q(i))
            end do
            !$omp enddo
            !$omp end parallel
        end if
    end subroutine advec_upwind_step
end module sample_1d_advection_step_module

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! 移流メインモジュール
module sample_1d_advection_main_module
    implicit none
    use sample_1d_advection_step_module

contains
    ! モジュール内のサブルーチン定義
    subroutine advec_upwind(nt, nx, dt, dx, u, q_init, q_final)
        implicit none

        ! 引数の属性設定
        integer, intent(in) :: nt, nx
        real, intent(in) :: dt, dx, u
        real, intent(in) :: q_init(nx)
        real, intent(out) :: q_final(nx)

        ! 濃度関数配列のメモリ確保
        real, allocatable :: q(nx), q_new(nx)
        allocate(q(nx), q_new(nx))

        ! 初期条件設定
        q = q_init

        ! ユーティリティ変数宣言
        integer :: i
        
        ! nt回の移流計算
        do i = 1, nt
            call advec_upwind_step(nx, dt, dx, u, q, q_new)
            q = q_new
        end do

        ! 最終結果を代入
        q_final = q

        ! メモリ解放
        deallocate(q, q_new)
    end subroutine advec_upwind
end module sample_1d_advection_main_module




    