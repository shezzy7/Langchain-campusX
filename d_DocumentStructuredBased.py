# Document Structured Based splitting refers to spliting data which is present not in plane text like any file that contains code in any programming language will not be a plain text.So to split such kind of data we need to use from_language method of RecursiveCharacterTextSplitter and inside this splitter we also have to tell that which language we are going to use(python etc)

from langchain.text_splitter import RecursiveCharacterTextSplitter , Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

text = """
class Book:
    def __init__(self, title, author, copies=1):
        self.title = title
        self.author = author
        self.copies = copies

    def __str__(self):
        return f"{self.title} by {self.author} ({self.copies} copies available)"


class Library:
    def __init__(self, name):
        self.name = name
        self.books = []

    def add_book(self, book):
        for b in self.books:
            if b.title == book.title and b.author == book.author:
                b.copies += book.copies
                return
        self.books.append(book)

    def display_books(self):
        if not self.books:
            print("No books available in the library.")
        for book in self.books:
            print(book)

    def borrow_book(self, title):
        for book in self.books:
            if book.title.lower() == title.lower() and book.copies > 0:
                book.copies -= 1
                print(f"You have borrowed: {book.title}")
                return
        print("Book not available.")

    def return_book(self, title):
        for book in self.books:
            if book.title.lower() == title.lower():
                book.copies += 1
                print(f"Returned: {book.title}")
                return
        print("This book doesn't belong to this library.")


# Test the system
lib = Library("City Library")
lib.add_book(Book("1984", "George Orwell", 3))
lib.add_book(Book("Python Programming", "John Zelle", 2))

lib.display_books()
lib.borrow_book("1984")
lib.display_books()
lib.return_book("1984")
lib.display_books()

"""

data = splitter.split_text(text)

print(data[0]) 