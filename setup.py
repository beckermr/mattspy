from setuptools import setup, find_packages

setup(
    name='mattspy',
    description="Matt's python utils",
    author='Matthew R. Becker',
    author_email='becker.mr@gmail.com',
    license="BSD-3-Clause",
    url='https://github.com/beckermr/mattspy',
    packages=find_packages(),
    include_package_data=True,
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
    entry_points={
        'console_scripts': [
            'condor-exec-run-pickled-task = mattspy.condor_exec_run:run_pickled_task',
        ],
    },
)
