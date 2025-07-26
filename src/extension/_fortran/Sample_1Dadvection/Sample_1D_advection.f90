! Sample fortran extension
! 1D advection equation simulator for HPC cpu node 
! Using OpenMP for parallelization

! モジュールの定義
module sample_1d_advection_module
    implicit none
    
contains
    ! モジュール内のサブルーチン定義
    subroutine advec_upwind_step(nx, dt, dx, u, q, q_new)
        implicit none

        ! 引数の属性設定
        real, intent(in) :: dt, dx, u
        integer, intent(in) :: nx
        real, intent(in) :: q(:)
        real, intent(out) :: q_new(:)

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
end module sample_1d_advection_module

