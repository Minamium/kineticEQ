! Sample fortran extension
! 1D advection equation simulator for HPC cpu node 
! Using OpenMP for parallelization

! 移流計算1ステップ
module sample_1d_advection_step_module
    implicit none
    
contains
    ! モジュール内のサブルーチン定義
    subroutine advec_upwind_step(nx, dt, dx, u, q, q_new)
        implicit none

        ! 引数の属性設定 - F2PY互換のためdouble precisionを使用
        double precision, intent(in) :: dt, dx, u
        integer, intent(in) :: nx
        double precision, intent(in) :: q(nx)
        double precision, intent(out) :: q_new(nx)

        ! ローカル変数
        double precision :: c
        integer :: i

        ! CFL数の計算
        c = u * dt / dx

        ! 一次風上差分（周期境界条件）
        if (u >= 0.0d0) then
            ! 正の流速：左から右へ
            q_new(1) = q(1) - c * (q(1) - q(nx))
            !$omp parallel
            !$omp do private(i)
            do i = 2, nx
                q_new(i) = q(i) - c * (q(i) - q(i - 1))
            end do
            !$omp enddo
            !$omp end parallel
        else
            ! 負の流速：右から左へ
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

! メイン移流計算
module sample_1d_advection_main_module
    use sample_1d_advection_step_module
    implicit none
    
contains
    subroutine advec_upwind(nt, nx, dt, dx, u, q_init, q_final)
        implicit none

        ! 引数の属性設定 - F2PY互換のためdouble precisionを使用
        integer, intent(in) :: nt, nx
        double precision, intent(in) :: dt, dx, u
        double precision, intent(in) :: q_init(nx)
        double precision, intent(out) :: q_final(nx)

        ! ローカル変数
        double precision, allocatable :: q(:), q_new(:)
        integer :: i

        ! 配列のメモリ確保
        allocate(q(nx), q_new(nx))

        ! 初期値のコピー
        q = q_init

        ! 時間ステップのループ
        do i = 1, nt
            call advec_upwind_step(nx, dt, dx, u, q, q_new)
            q = q_new
        end do

        ! 結果のコピー
        q_final = q

        ! メモリ解放
        deallocate(q, q_new)
    end subroutine advec_upwind
end module sample_1d_advection_main_module