import setuptools

setuptools.setup(
    name='jmt2',
    version='0.1.0.dev0',
    license='MIT',
    author='joonas',
    author_email='joonas.yoon@gmail.com',
    description='PyTorch Model Trainer',
    long_description=open('README.md', encoding='utf-8').read(),
    url='https://github.com/joonas-yoon/jmt',
    packages=setuptools.find_packages(),
    classifiers=[
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=['pytorch', 'deep learning', 'trainer'],
    python_requires='>=3.6.0',
)
