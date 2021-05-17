
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='paddle-lpips',  
     version='0.1.2',
     author="AgentMaker",
     author_email="agentmaker@163.com",
     description="PaddlePaddle version of LPIPS Similarity metric",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/AgentMaker/Paddle-PerceptualSimilarity",
     packages=['paddle_lpips', 'paddle_lpips/models'],
     package_data={'paddle_lpips': ['weights/v0.0/*.pdparams','weights/v0.1/*.pdparams']},
     include_package_data=True,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
     ],
 )
