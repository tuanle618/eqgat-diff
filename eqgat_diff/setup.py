from setuptools import setup

setup(
    use_scm_version={
        "version_scheme": "post-release",
        "write_to": "e3moldiffusion/_version.py",
    },
    setup_requires=["setuptools_scm"],
)