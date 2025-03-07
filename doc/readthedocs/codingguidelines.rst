.. _coding-guidelines:

Coding Guidelines
==================

General Guidelines regarding Coding in C++
--------------------------------------------

Current C++ standard in |FOURC| is C++20. Hence, use of C++20 features is encouraged.
Avoid define flags as much as possible, because they complicate testing and, thus, lead to untested code.
Avoid and actively resolve header-in-header inclusion to speed up compilation time. Use forward declarations instead.
Do not use ``using ... / typedef`` statements at unscoped level.

Really use the refactoring methods provided by your IDE to write clean code.
For example: If renaming variables only takes seconds,
you can use it as many times as needed until you find a name that makes your code easy to read and understand.


|FOURC|-specific Design Guidelines
------------------------------------

**Preamble:** |FOURC| is a legacy code.
Large parts were written pre-C++11 and are not modern C++.
In addition, code review was not widely practiced back in these days.
Thus, it is recommended to critically examine the code and techniques while working on the code base and refer to the general guidelines above.

The following guidelines are especially relevant for the current state of |FOURC|:

- If necessary, use smart pointers for their memory management capabilities, e.g., ``std::shared_ptr``, ``std::unique_ptr``
  (you may find more information on passing smart pointers `here <https://www.modernescpp.com/index.php/c-core-guidelines-passing-smart-pointer/>`_
- by default, pass parameters by const reference and only expose smart pointers if memory management is necessary
- Prefer parameter container classes over ``Teuchos::ParameterList`` for passing parameters among element routines
- Make new code const-correct and fix old code in that regard when working on it.
- Implement new features in the new structural time integration. Work towards migrating existing capabilities to the new structural time integration.
- Test your code! Refer to our documentation about :ref:`Testing<4Ctesting>`.


|FOURC|-specific Naming Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Namespaces** use CamelCase: ``LinAlg::``
- **Class names** / **function** names / **enum types** use camel case, starting with a capital letter:
  ``ClassToDoSomething / FunctionToPerformAnOperation()``
- **Variables** and **enum values** use

    - snake_case, i.e. all small letters separated by underscores, e.g. ``variable_with_descriptive_name``
    - camelCase staring with a small letter, e.g. ``variableWithDescriptiveName``

- **Class members** end with an underscore: ``variable_``
- **Define flags** are in all caps: ``DEBUG``

Variable names must not be just a single letter, because they are impossible to find in a global search operation.
(Exception: loop indices such as i, j, but remember that even loop indices could/should have descriptive names.)
