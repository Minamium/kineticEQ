! Sample fortran extension
! 1D advection equation simulator for HPC cpu node 
! Using OpenMP for parallelization

! Module definition
module sample_1d_advection_module
    implicit none
    
contains
    ! Subroutine definition within the module
    subroutine advec_upwind_step(nx, dt, dx, u, q, q_new)
        implicit none

        ! Argument attribute settings
        integer, intent(in) :: nx
        real, intent(in) :: dt, dx, u
        real, intent(in) :: q(nx)
        real, intent(out) :: q_new(nx)

        ! Utility variable declarations
        integer :: i
        real :: c

        ! CFL number calculation
        c = u * (dt / dx)

        ! Advection calculation (MP parallelization) Boundary condition: Periodic boundary, Difference scheme: First-order upwind difference
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
