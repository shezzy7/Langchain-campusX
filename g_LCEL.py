# LCEL stands for langchain expression language.We see that RunnableSequence is used mostly when it comes to chaining.So instead of writing RunnableSequence all the times we can use | operator for combining mulitple runnables to form a chain.

# for example if we have three components template1,model, and parser.And we have to combine them to form a chain.
# We can do so by this method -> RunnableSequence(template1,model,parser)
# And we also do this by this mthod -> template | model | parser
