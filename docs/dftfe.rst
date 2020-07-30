Key DFT-FE Objects
******************

Top-Level Classes
-----------------

.. doxygenclass:: dftfe::dftClass
   :members:

.. doxygenclass:: dftfe::kohnShamDFTOperatorClass
   :members:

.. doxygenclass:: dftfe::kohnShamDFTOperatorCUDAClass
   :members:

.. doxygenclass:: dftfe::forceClass
   :members:

.. doxygenclass:: dftfe::geoOptIon
   :members:

.. doxygenclass:: dftfe::geoOptCell
   :members:

.. doxygenclass:: dftfe::symmetryClass
   :members:

.. doxygenclass:: dftfe::molecularDynamics
   :members:

Advanced Classes
----------------

.. doxygenclass:: dftfe::operatorDFTClass
   :members:

.. we should call this the `choir` method:
.. doxygenclass:: dftfe::chebyshevOrthogonalizedSubspaceIterationSolver
   :members:

.. doxygenclass:: dftfe::chebyshevOrthogonalizedSubspaceIterationSolverCUDA
   :members:

Advanced Functions
------------------

.. doxygennamespace:: dftfe::linearAlgebraOperations
   :members:

.. doxygennamespace:: dftfe::linearAlgebraOperationsCUDA
   :members:

.. We should add -> doxygenfunction:: dftfe::stridedCopyToBlockKernel

