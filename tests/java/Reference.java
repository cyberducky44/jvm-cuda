import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;

public final class Reference {
    public static void main(String[] args) throws Exception {
        int count = Integer.parseInt(args[2]);
        Class<?>[] types = new Class<?>[count];
        Arrays.fill(types, int.class);
        Method method = Class.forName(args[0]).getDeclaredMethod(args[1], types);
        BufferedReader input = new BufferedReader(new InputStreamReader(System.in));
        String line;
        while ((line = input.readLine()) != null) {
            Object[] values = new Object[count];
            String[] parts = line.trim().split("\\s+");
            for (int i = 0; i < count; i++) values[i] = Integer.valueOf(parts[i]);
            try {
                System.out.println(method.invoke(null, values));
            } catch (InvocationTargetException error) {
                System.out.println(error.getCause().getClass().getName());
            }
        }
    }
}
